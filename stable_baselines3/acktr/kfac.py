"""
Kronecker-Factored Approximate Curvature (K-FAC) Optimizer for PyTorch

Based on the paper:
"Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation"
https://arxiv.org/abs/1708.05144

This implementation is adapted from:
- https://github.com/openai/baselines (TensorFlow version)
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail (PyTorch version)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class KFACOptimizer(optim.Optimizer):
    """
    K-FAC optimizer for natural gradient descent.

    :param model: Neural network model
    :param lr: Learning rate
    :param momentum: Momentum coefficient
    :param stat_decay: Moving average decay for statistics
    :param damping: Damping parameter for numerical stability
    :param kl_clip: KL divergence clipping threshold
    :param weight_decay: L2 penalty coefficient
    :param update_freq: Frequency of updating second-order statistics
    :param cold_start_steps: Number of initial steps to use standard SGD before K-FAC
    :param cold_start_lr: Learning rate during cold start phase
    :param max_grad_norm: Maximum gradient norm for clipping
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.25,
        momentum: float = 0.9,
        stat_decay: float = 0.99,
        damping: float = 1e-3,
        kl_clip: float = 0.001,
        weight_decay: float = 0,
        update_freq: int = 1,
        cold_start_steps: int = 10,
        cold_start_lr: float | None = None,
        max_grad_norm: float | None = None,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(model.parameters(), defaults)

        self.model = model
        self.stat_decay = stat_decay
        self.damping = damping
        self.kl_clip = kl_clip
        self.update_freq = update_freq
        self.cold_start_steps = cold_start_steps
        self.cold_start_lr = cold_start_lr if cold_start_lr is not None else lr
        self.max_grad_norm = max_grad_norm

        self.steps = 0
        self.known_modules: dict[nn.Module, str] = {}
        self.modules: list[nn.Module] = []

        # Cache for identity matrices to avoid recomputing them
        # Key is (size, device_str, dtype_str) for hashability
        self.identity_cache: dict[tuple[int, str, str], torch.Tensor] = {}

        # Initialize Fisher information matrices for each module
        self.activations: dict[nn.Module, torch.Tensor | None] = {}  # Store activations
        self.gradients: dict[nn.Module, torch.Tensor | None] = {}  # Store gradients
        self.m_aa: dict[nn.Module, torch.Tensor | None] = {}  # Fisher for activations
        self.m_gg: dict[nn.Module, torch.Tensor | None] = {}  # Fisher for gradients

        self._prepare_model()

    def _prepare_model(self) -> None:
        """
        Extract supported layers from the model and initialize their statistics.
        Supported layers: Linear, Conv2d
        """
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in ["Linear", "Conv2d"]:
                self.modules.append(module)
                self.known_modules[module] = classname
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

        for module in self.modules:
            self.m_aa[module] = None
            self.m_gg[module] = None
            self.activations[module] = None
            self.gradients[module] = None

    def _save_input(self, module: nn.Module, input: tuple[torch.Tensor, ...]) -> None:
        """Hook to save layer inputs for computing Fisher information."""
        if torch.is_grad_enabled() and self.steps % self.update_freq == 0:
            classname = self.known_modules[module]
            aa = input[0].detach()
            if classname == "Linear":
                assert aa.dim() == 2, f"Expected 2D tensor for Linear, got {aa.dim()}D"
            elif classname == "Conv2d":
                assert aa.dim() == 4, f"Expected 4D tensor for Conv2d, got {aa.dim()}D"
            # Store activations directly; further processing done in _update_fisher_stats
            self.activations[module] = aa

    def _save_grad_output(
        self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        """Hook to save layer gradient outputs for computing Fisher information."""
        if self.steps % self.update_freq == 0:
            classname = self.known_modules[module]
            gg = grad_output[0].detach()
            if classname == "Linear":
                assert gg.dim() == 2, f"Expected 2D tensor for Linear, got {gg.dim()}D"
            elif classname == "Conv2d":
                assert gg.dim() == 4, f"Expected 4D tensor for Conv2d, got {gg.dim()}D"
            # Store gradients directly; further processing done in _update_fisher_stats
            self.gradients[module] = gg

    def _update_fisher_stats(self) -> None:
        """Update Fisher information matrices using stored activations and gradients."""
        for module in self.modules:
            if self.activations[module] is not None and self.gradients[module] is not None:
                classname = self.known_modules[module]

                aa = self.activations[module]
                gg = self.gradients[module]

                # Compute Cov_A and Cov_G based on layer type
                if classname == "Linear":
                    # aa: (batch_size, in_features)
                    # Add bias term
                    if module.bias is not None:
                        aa = torch.cat([aa, aa.new_ones(aa.size(0), 1)], 1)
                    cov_a = aa.t() @ aa / aa.size(0)
                    # gg: (batch_size, out_features)
                    cov_g = gg.t() @ gg / gg.size(0)
                elif classname == "Conv2d":
                    # aa: (batch_size, in_channels, height, width)
                    # aa_unfold: (batch_size, in_channels * kernel_height * kernel_width, num_patches)
                    aa_unfold = F.unfold(aa, module.kernel_size, padding=module.padding, stride=module.stride)
                    # Reshape to (batch_size * num_patches, in_channels * kernel_height * kernel_width)
                    aa_unfold = aa_unfold.permute(0, 2, 1).contiguous().view(-1, aa_unfold.size(1))
                    # Add bias term
                    if module.bias is not None:
                        aa_unfold = torch.cat([aa_unfold, aa_unfold.new_ones(aa_unfold.size(0), 1)], 1)
                    cov_a = aa_unfold.t() @ aa_unfold / aa_unfold.size(0)
                    # gg: (batch_size, out_channels, out_height, out_width)
                    # Reshape to (batch_size * out_height * out_width, out_channels)
                    gg_reshaped = gg.permute(0, 2, 3, 1).contiguous().view(-1, gg.size(1))
                    cov_g = gg_reshaped.t() @ gg_reshaped / gg_reshaped.size(0)

                # Update moving averages
                if self.m_aa[module] is None:
                    self.m_aa[module] = cov_a
                else:
                    self.m_aa[module].mul_(self.stat_decay).add_(cov_a, alpha=1 - self.stat_decay)

                if self.m_gg[module] is None:
                    self.m_gg[module] = cov_g
                else:
                    self.m_gg[module].mul_(self.stat_decay).add_(cov_g, alpha=1 - self.stat_decay)

    def _get_identity(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get cached identity matrix or create a new one.

        :param size: Size of the identity matrix
        :param device: Device to create the matrix on
        :param dtype: Data type of the matrix
        :return: Identity matrix
        """
        # Use string representations for hashability
        key = (size, str(device), str(dtype))
        if key not in self.identity_cache:
            self.identity_cache[key] = torch.eye(size, device=device, dtype=dtype)
        return self.identity_cache[key]

    def _clip_gradients(self) -> None:
        """Clip gradients to the maximum norm if specified."""
        total_norm = 0.0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)

    def _sgd_momentum_step(self) -> None:
        """Perform a standard SGD momentum step."""
        # Clip gradients before the update if specified
        if self.max_grad_norm is not None:
            self._clip_gradients()

        # Standard SGD momentum update
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param.data)

                v = state["momentum_buffer"]
                v.mul_(group["momentum"]).add_(grad)

                if group["weight_decay"] != 0:
                    param.data.add_(param.data, alpha=-group["weight_decay"] * group["lr"])
                param.data.add_(v, alpha=-group["lr"])

    def _apply_fisher_preconditioned_grad(
        self,
        module: nn.Module,
        grad: torch.Tensor,
        m_aa: torch.Tensor,
        m_gg: torch.Tensor,
        state: dict,
        momentum: float,
    ) -> torch.Tensor:
        """Apply K-FAC preconditioning to the gradient."""
        damping = self.damping
        g_reshape = grad.data
        
        if self.known_modules[module] == "Linear":
            # For Linear: weight is (out_features, in_features)
            if module.bias is not None and g_reshape.size(1) == m_aa.size(0) - 1:
                # Pad for bias
                if module.bias.grad is not None:
                    bias_grad = module.bias.grad.data
                else:
                    bias_grad = torch.zeros_like(module.bias)
                g_reshape = torch.cat([g_reshape, bias_grad.unsqueeze(1)], 1)
        elif self.known_modules[module] == "Conv2d":
            # For Conv2d: weight is (out_channels, in_channels, kH, kW)
            g_reshape = g_reshape.view(g_reshape.size(0), -1)
            if module.bias is not None and g_reshape.size(1) == m_aa.size(0) - 1:
                if module.bias.grad is not None:
                    bias_grad = module.bias.grad.data
                else:
                    bias_grad = torch.zeros_like(module.bias)
                g_reshape = torch.cat([g_reshape, bias_grad.unsqueeze(1)], 1)

        # SVD of Fisher matrices
        d_a, Q_a = torch.linalg.eigh(m_aa + damping * self._get_identity(m_aa.size(0), m_aa.device, m_aa.dtype))
        d_g, Q_g = torch.linalg.eigh(m_gg + damping * self._get_identity(m_gg.size(0), m_gg.device, m_gg.dtype))
        
        # Invert eigenvalues with damping
        d_a_inv = 1.0 / d_a
        d_g_inv = 1.0 / d_g
        
        # Compute natural gradient
        v1 = Q_g.t() @ g_reshape @ Q_a
        v2 = v1 / (d_g_inv.unsqueeze(1) @ d_a_inv.unsqueeze(0) + damping)
        v3 = Q_g @ v2 @ Q_a.t()

        # Apply momentum
        v = state["momentum_buffer"]
        v.mul_(momentum).add_(v3)

        # Apply natural gradient
        if self.known_modules[module] == "Linear":

    def _kfac_step(self) -> None:
        """Perform a K-FAC natural gradient step."""
        # Update Fisher matrices from stored activations/gradients
        if (self.steps - self.cold_start_steps) % self.update_freq == 0:
            self._update_fisher_stats()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                # Find the module this parameter belongs to
                module = None
                is_bias = False
                for m in self.modules:
                    if param is m.weight:
                        module = m
                        is_bias = False
                        break
                    elif hasattr(m, "bias") and m.bias is not None and param is m.bias:
                        module = m
                        is_bias = True
                        break

                # Bias updates are handled together with weights in K-FAC
                if is_bias:
                    continue

                # Standard grad dient descent for parameters not in known modules
                if module is None:
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param.data)

                    v = state["momentum_buffer"]
                    v.mul_(group["momentum"]).add_(grad)

                    if group["weight_decay"] != 0:
                        param.data.add_(param.data, alpha=-group["weight_decay"] * group["lr"])
                    param.data.add_(v, alpha=-group["lr"])
                    continue

                # Apply natural gradient update for weights
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param.data)

                # v = state["momentum_buffer"]
                # v.mul_(group["momentum"]).add_(grad)

                # Compute natural gradient
                assert (self.m_aa[module] is not None and self.m_gg[module] is not None), "Fisher information matrices have not been initialized."
                m_gg = self.m_gg[module]
                m_aa = self.m_aa[module]

                # Add damping for numerical stability using cached identity matrices
                # m_gg_damp = m_gg + self.damping * self._get_identity(m_gg.size(0), m_gg.device, m_gg.dtype)
                # m_aa_damp = m_aa + self.damping * self._get_identity(m_aa.size(0), m_aa.device, m_aa.dtype)

                # Compute natural gradient using Kronecker-factored preconditioner
                # Natural gradient = inv(G) @ grad @ inv(A)
                # where G is gradient covariance and A is activation covariance
                if self.known_modules[module] == "Linear":
                    # For Linear: weight is (out_features, in_features)
                    # With bias handling in activation stats
                    g_reshape = v.data
                    if module.bias is not None and g_reshape.size(1) == m_aa_damp.size(0) - 1:
                        # Pad for bias
                        if module.bias.grad is not None:
                            bias_grad = module.bias.grad.data
                        else:
                            bias_grad = torch.zeros_like(module.bias)
                        g_reshape = torch.cat([g_reshape, bias_grad.unsqueeze(1)], 1)

                    # Apply Kronecker-factored preconditioner
                    inv_gg = torch.linalg.inv(m_gg_damp)
                    inv_aa = torch.linalg.inv(m_aa_damp)
                    natural_grad = torch.linalg.multi_dot([inv_gg, g_reshape, inv_aa])

                    # Extract weight update
                    if module.bias is not None and natural_grad.size(1) > v.size(1):
                        param.data.add_(natural_grad[:, :-1], alpha=-group["lr"])
                        # Update bias
                        if module.bias.grad is not None:
                            module.bias.data.add_(natural_grad[:, -1], alpha=-group["lr"])
                    else:
                        param.data.add_(natural_grad, alpha=-group["lr"])

                elif self.known_modules[module] == "Conv2d":
                    # For Conv2d: simplified version treating as matrix
                    g_reshape = v.data.view(v.size(0), -1)
                    if module.bias is not None and g_reshape.size(1) == m_aa_damp.size(0) - 1:
                        if module.bias.grad is not None:
                            bias_grad = module.bias.grad.data
                        else:
                            bias_grad = torch.zeros_like(module.bias)
                        g_reshape = torch.cat([g_reshape, bias_grad.unsqueeze(1)], 1)

                    inv_gg = torch.linalg.inv(m_gg_damp)
                    inv_aa = torch.linalg.inv(m_aa_damp)
                    natural_grad = torch.linalg.multi_dot([inv_gg, g_reshape, inv_aa])

                    if module.bias is not None and natural_grad.size(1) > v.view(v.size(0), -1).size(1):
                        param.data.add_(natural_grad[:, :-1].view_as(v), alpha=-group["lr"])
                        if module.bias.grad is not None:
                            module.bias.data.add_(natural_grad[:, -1], alpha=-group["lr"])
                    else:
                        param.data.add_(natural_grad.view_as(v), alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:  # noqa: C901
        """
        Perform a single optimization step.
        Use standard SGD momentum during cold start phase, then switch to K-FAC natural gradient.

        :param closure: A closure that reevaluates the model and returns the loss
        :return: Loss if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.steps < self.cold_start_steps:
            # Use standard SGD momentum during cold start phase
            self._sgd_momentum_step()
        else:
            # Use K-FAC natural gradient step
            self._kfac_step()

        self.steps += 1
        return loss
