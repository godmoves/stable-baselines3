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
        self.param_to_module: dict[torch.Tensor, nn.Module] = {}

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
                self.param_to_module[module.weight] = module
                if module.bias is not None:
                    self.param_to_module[module.bias] = module
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
            a = input[0].detach()
            if classname == "Linear":
                assert a.dim() == 2, f"Expected 2D tensor for Linear, got {a.dim()}D"
            elif classname == "Conv2d":
                assert a.dim() == 4, f"Expected 4D tensor for Conv2d, got {a.dim()}D"
            # Store activations directly; further processing done in _update_fisher_stats
            self.activations[module] = a

    def _save_grad_output(
        self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        """Hook to save layer gradient outputs for computing Fisher information."""
        if self.steps % self.update_freq == 0:
            classname = self.known_modules[module]
            g = grad_output[0].detach()
            if classname == "Linear":
                assert g.dim() == 2, f"Expected 2D tensor for Linear, got {g.dim()}D"
            elif classname == "Conv2d":
                assert g.dim() == 4, f"Expected 4D tensor for Conv2d, got {g.dim()}D"
            # Store gradients directly; further processing done in _update_fisher_stats
            self.gradients[module] = g

    def _update_fisher_stats(self) -> None:
        """
        Update Fisher information matrices using stored activations and gradients.
        Here we assume the loss is averaged over the batch.
        """
        for module in self.modules:
            if self.activations[module] is not None and self.gradients[module] is not None:
                classname = self.known_modules[module]

                a = self.activations[module]
                g = self.gradients[module]

                # Compute Cov_A and Cov_G based on layer type
                if classname == "Linear":
                    # a: (batch_size, in_features)
                    B = a.size(0)  # batch_size
                    if module.bias is not None:
                        a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)
                    cov_a = a.t() @ a / B
                    # g: (batch_size, out_features)
                    g_scaled = g * B  # Scale gradient by batch size
                    cov_g = g_scaled.t() @ g_scaled / B
                elif classname == "Conv2d":
                    # a: (batch_size, in_channels, height, width)
                    # a_unfold: (batch_size, in_channels * kernel_height * kernel_width, num_patches)
                    a_unfold = F.unfold(a, module.kernel_size, dilation=module.dilation, padding=module.padding, stride=module.stride)
                    B = a_unfold.size(0)  # batch_size
                    T = a_unfold.size(2)  # num_patches = out_height * out_width
                    # Reshape to (batch_size * num_patches, in_channels * kernel_height * kernel_width)
                    a_unfold = a_unfold.permute(0, 2, 1).contiguous().view(-1, a_unfold.size(1))
                    a_unfold = a_unfold / T
                    if module.bias is not None:
                        a_unfold = torch.cat(
                            [a_unfold, a_unfold.new_ones(a_unfold.size(0), 1) / T],
                            dim=1,
                        )
                    cov_a = a_unfold.t() @ a_unfold / B
                    # g: (batch_size, out_channels, out_height, out_width)
                    # Reshape to (batch_size * out_height * out_width, out_channels)
                    g_2d = g.permute(0, 2, 3, 1).contiguous().view(-1, g.size(1))
                    g_2d = g_2d * T
                    g_2d = g_2d * B
                    cov_g = g_2d.t() @ g_2d / (B * T)

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

    def _get_momentum_buffer(self, param: torch.Tensor) -> torch.Tensor:
        """Get (and lazily initialize) the momentum buffer for a parameter."""
        param_state = self.state[param]
        if len(param_state) == 0:
            param_state["momentum_buffer"] = torch.zeros_like(param.data)
        return param_state["momentum_buffer"]

    def _clear_momentum_buffers(self) -> None:
        """Clear all momentum buffers."""
        for group in self.param_groups:
            for param in group["params"]:
                param_state = self.state[param]
                if "momentum_buffer" in param_state:
                    param_state["momentum_buffer"].zero_()

    def _sgd_momentum_step(self) -> None:
        """Perform a standard SGD momentum step."""
        # Clip gradients before the update if specified
        if self.max_grad_norm is not None:
            all_params = [p for group in self.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            
        # Standard SGD momentum update
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                v = self._get_momentum_buffer(param)
                # Standard / Polyak momentum update, v = momentum * v + grad
                v.mul_(group["momentum"]).add_(grad)

                # if group["weight_decay"] != 0:
                #     # Use decoupled weight decay as in AdamW
                #     param.data.add_(param.data, alpha=-group["weight_decay"] * group["lr"])
                param.data.add_(v, alpha=-group["lr"])

    def _apply_fisher_preconditioned_grad(
        self,
        grad: torch.Tensor,
        m_aa: torch.Tensor,
        m_gg: torch.Tensor,
        lam: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Apply K-FAC preconditioning to the gradient."""
        # SVD of Fisher matrices
        d_a, Q_a = torch.linalg.eigh(m_aa)
        d_g, Q_g = torch.linalg.eigh(m_gg)

        # Clip eigenvalues to avoid numerical issues
        d_a.clamp_(min=eps)
        d_g.clamp_(min=eps)

        # Compute natural gradient
        v1 = Q_g.t() @ grad @ Q_a
        v2 = v1 / (d_g.unsqueeze(1) @ d_a.unsqueeze(0) + lam)
        v3 = Q_g @ v2 @ Q_a.t()

        return v3

    def _kfac_step(self) -> None:
        """Perform a K-FAC natural gradient step."""
        # Update Fisher matrices from stored activations/gradients
        if (self.steps - self.cold_start_steps) % self.update_freq == 0:
            self._update_fisher_stats()

        kfac_update = {}
        vg_sum = 0.0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                # Effective learning rate considering momentum
                effective_lr = lr * (1 - momentum)

                # Find the module this parameter belongs to
                module = self.param_to_module.get(param, None)
                if module and getattr(module, "bias", None) is param:
                    # Bias updates are handled together with weights in K-FAC
                    continue

                # Standard grad dient descent for parameters not in known modules
                if module is None:
                    v = self._get_momentum_buffer(param)
                    v.mul_(momentum).add_(grad)

                    # if weight_decay != 0:
                    #     param.data.add_(param.data, alpha=-weight_decay * lr)
                    param.data.add_(v, alpha=-effective_lr)
                    continue

                # Apply natural gradient update for weights
                assert (self.m_aa[module] is not None and self.m_gg[module] is not None), "Fisher information matrices have not been initialized."
                g = grad.data
                m_aa = self.m_aa[module]
                m_gg = self.m_gg[module]
        
                # Handle bias by augmenting gradient
                if self.known_modules[module] == "Linear":
                    # For Linear: weight is (out_features, in_features)
                    if module.bias is not None and g.size(1) == m_aa.size(0) - 1:
                        # Pad for bias
                        if module.bias.grad is not None:
                            bias_grad = module.bias.grad.data
                        else:
                            bias_grad = torch.zeros_like(module.bias.data)
                        g = torch.cat([g, bias_grad.unsqueeze(1)], 1)
                elif self.known_modules[module] == "Conv2d":
                    # For Conv2d: weight is (out_channels, in_channels, kernel_height, kernel_width)
                    g = g.view(g.size(0), -1)
                    if module.bias is not None and g.size(1) == m_aa.size(0) - 1:
                        if module.bias.grad is not None:
                            bias_grad = module.bias.grad.data
                        else:
                            bias_grad = torch.zeros_like(module.bias.data)
                        g = torch.cat([g, bias_grad.unsqueeze(1)], 1)

                v = self._apply_fisher_preconditioned_grad(
                    g,
                    m_aa,
                    m_gg,
                    lam=self.damping + weight_decay,
                )

                # Split weight and bias updates
                if self.known_modules[module] == "Linear":
                    if module.bias is not None and v.size(1) == param.data.size(1) + 1:
                        weight_update = v[:, :-1]
                        bias_update = v[:, -1]
                    else:
                        weight_update = v
                        bias_update = None
                elif self.known_modules[module] == "Conv2d":
                    if module.bias is not None and v.size(1) == param.data.numel() + 1:
                        weight_update = v[:, :-1].view_as(param.data)
                        bias_update = v[:, -1]
                    else:
                        weight_update = v.view_as(param.data)
                        bias_update = None

                # Record the update
                kfac_update[param] = {
                    "module": module,
                    "weight_update": weight_update,
                    "bias_update": bias_update,
                    "lr": lr,
                    "momentum": momentum,
                    "effective_lr": effective_lr,
                    "weight_decay": weight_decay,
                }
                vg_sum += (v * g * lr * lr).sum().item()

        # Compute scaling factor for KL clipping, i.e., eta_max is fixed as 1.0
        scaling = torch.min(torch.tensor(1.0), torch.sqrt(self.kl_clip / (vg_sum + 1e-10)))
        for param, update_info in kfac_update.items():
            module = update_info["module"]
            weight_update = update_info["weight_update"]
            bias_update = update_info["bias_update"]
            lr = update_info["lr"]
            effective_lr = update_info["effective_lr"]
            momentum = update_info["momentum"]

            # Update momentum buffer
            v = self._get_momentum_buffer(param)
            v.mul_(momentum).add_(weight_update, alpha=effective_lr * scaling)
            param.data.add_(v, alpha=-1.0)

            # Update bias if applicable
            if bias_update is not None and module.bias is not None:
                v_bias = self._get_momentum_buffer(module.bias)
                v_bias.mul_(momentum).add_(bias_update, alpha=effective_lr * scaling)
                module.bias.data.add_(v_bias, alpha=-1.0)

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
            # Clear momentum buffers if transitioning from cold start
            if self.steps == self.cold_start_steps:
                self._clear_momentum_buffers()

            # Use K-FAC natural gradient step
            self._kfac_step()

        self.steps += 1
        return loss
