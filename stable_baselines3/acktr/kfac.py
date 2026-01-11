"""
Kronecker-Factored Approximate Curvature (K-FAC) Optimizer for PyTorch

Based on the paper:
"Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation"
https://arxiv.org/abs/1708.05144

This implementation is adapted from:
- https://github.com/openai/baselines (TensorFlow version)
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail (PyTorch version)
"""

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


class KFACOptimizer(optim.Optimizer):
    """
    K-FAC optimizer for natural gradient descent.

    :param model: Neural network model
    :param lr: Learning rate
    :param momentum: Momentum coefficient
    :param stat_decay: Moving average decay for statistics
    :param kl_clip: KL divergence clipping threshold
    :param damping: Damping parameter for numerical stability
    :param weight_decay: L2 penalty coefficient
    :param update_freq: Frequency of updating second-order statistics
    :param alpha: Running average parameter (unused, kept for API compatibility)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.25,
        momentum: float = 0.9,
        stat_decay: float = 0.99,
        kl_clip: float = 0.001,
        damping: float = 1e-2,
        weight_decay: float = 0,
        update_freq: int = 1,
        alpha: float = 0.95,
    ):
        defaults = dict(lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay)
        super().__init__(model.parameters(), defaults)

        self.known_modules: dict[nn.Module, str] = {}
        self.modules: list[nn.Module] = []
        self.model = model
        self.stat_decay = stat_decay
        self.damping = damping
        self.update_freq = update_freq

        self.steps = 0
        # Cache for identity matrices to avoid recomputing them
        self.identity_cache: dict[int, torch.Tensor] = {}
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

        # Initialize Fisher information matrices for each module
        self.m_aa: dict[nn.Module, torch.Tensor | None] = {}  # Fisher for activations
        self.m_gg: dict[nn.Module, torch.Tensor | None] = {}  # Fisher for gradients
        self.activations: dict[nn.Module, torch.Tensor | None] = {}  # Store activations
        self.gradients: dict[nn.Module, torch.Tensor | None] = {}  # Store gradients

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

            # Compute statistics based on layer type
            if classname == "Linear":
                # For Linear layers, flatten if needed and add bias term
                if aa.dim() > 2:
                    aa = aa.view(aa.size(0), -1)
                # Add bias term (homogeneous coordinates)
                if module.bias is not None:
                    aa = torch.cat([aa, aa.new_ones(aa.size(0), 1)], 1)
                self.activations[module] = aa

            elif classname == "Conv2d":
                # For Conv2d, we compute spatial mean of activations
                # This is a simplification - full K-FAC would extract patches
                batch_size = aa.size(0)
                channels = aa.size(1)

                # Reshape and compute mean over spatial dimensions
                aa = aa.view(batch_size, channels, -1).mean(2)

                # Add bias term
                if module.bias is not None:
                    aa = torch.cat([aa, aa.new_ones(batch_size, 1)], 1)
                self.activations[module] = aa

    def _save_grad_output(
        self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        """Hook to save layer gradient outputs for computing Fisher information."""
        if self.steps % self.update_freq == 0:
            classname = self.known_modules[module]
            gg = grad_output[0].detach()

            if classname == "Linear":
                # For Linear layers
                if gg.dim() > 2:
                    gg = gg.view(gg.size(0), -1)
                self.gradients[module] = gg

            elif classname == "Conv2d":
                # For Conv2d, spatial mean
                batch_size = gg.size(0)
                channels = gg.size(1)
                # Take mean over spatial dimensions
                gg = gg.view(batch_size, channels, -1).mean(2)
                self.gradients[module] = gg

    def _update_fisher(self) -> None:
        """Update Fisher information matrices using stored activations and gradients."""
        for module in self.modules:
            if self.activations[module] is not None and self.gradients[module] is not None:
                aa = self.activations[module]
                gg = self.gradients[module]

                # Compute covariance matrices
                aa_t = torch.mm(aa.t(), aa) / aa.size(0)
                gg_t = torch.mm(gg.t(), gg) / gg.size(0)

                # Update moving averages
                if self.m_aa[module] is None:
                    self.m_aa[module] = aa_t
                else:
                    self.m_aa[module] = self.stat_decay * self.m_aa[module] + (1 - self.stat_decay) * aa_t

                if self.m_gg[module] is None:
                    self.m_gg[module] = gg_t
                else:
                    self.m_gg[module] = self.stat_decay * self.m_gg[module] + (1 - self.stat_decay) * gg_t

    def _get_identity(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get cached identity matrix or create a new one.

        :param size: Size of the identity matrix
        :param device: Device to create the matrix on
        :param dtype: Data type of the matrix
        :return: Identity matrix
        """
        key = (size, device, dtype)
        if key not in self.identity_cache:
            self.identity_cache[key] = torch.eye(size, device=device, dtype=dtype)
        return self.identity_cache[key]

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:  # noqa: C901
        """
        Perform a single optimization step using natural gradient.

        :param closure: A closure that reevaluates the model and returns the loss
        :return: Loss if closure is provided, None otherwise
        """
        # Register hooks if first step
        if self.steps == 0:
            self._register_hooks()

        # Update Fisher matrices from stored activations/gradients
        if self.steps % self.update_freq == 0:
            self._update_fisher()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

                if module is None or is_bias:
                    # Standard gradient descent for parameters not in known modules or bias
                    # Bias updates are handled together with weights in K-FAC
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

                v = state["momentum_buffer"]
                v.mul_(group["momentum"]).add_(grad)

                # Compute natural gradient
                if self.m_aa[module] is not None and self.m_gg[module] is not None:
                    m_gg = self.m_gg[module]
                    m_aa = self.m_aa[module]

                    # Add damping for numerical stability using cached identity matrices
                    m_gg_damp = m_gg + self.damping * self._get_identity(m_gg.size(0), m_gg.device, m_gg.dtype)
                    m_aa_damp = m_aa + self.damping * self._get_identity(m_aa.size(0), m_aa.device, m_aa.dtype)

                    # Compute natural gradient using Kronecker-factored preconditioner
                    # Natural gradient = inv(G) @ grad @ inv(A)
                    # where G is gradient covariance and A is activation covariance
                    try:
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

                    except (RuntimeError, torch.linalg.LinAlgError):
                        # Fallback to standard momentum if matrix inversion fails
                        # This can happen if the Fisher matrix is singular or ill-conditioned
                        param.data.add_(v, alpha=-group["lr"])
                else:
                    # Fallback to standard momentum update if Fisher not computed yet
                    param.data.add_(v, alpha=-group["lr"])

        self.steps += 1
        return loss

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on all supported modules."""
        for module in self.modules:
            module.register_forward_pre_hook(self._save_input)
            module.register_full_backward_hook(self._save_grad_output)
