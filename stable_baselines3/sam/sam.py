"""SAM (Sharpness-Aware Minimization) optimizer.

Reference:
  - Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization".

This is a lightweight PyTorch implementation that wraps a "base" optimizer
(e.g., SGD/Adam) and performs the two-step SAM update.

Typical usage:

  optimizer = SAM(model.parameters(), torch.optim.SGD, lr=1e-3, momentum=0.9)

  def closure():
      optimizer.zero_grad()
      loss = loss_fn(model(x), y)
      loss.backward()
      return loss

  loss = optimizer.step(closure)

You can also call the two phases explicitly via `first_step()` and
`second_step()`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.

    SAM solves
    $\\min_w \\max_{\\|\\epsilon\\|_2 \\le \\rho} L(w + \\epsilon)$

    by approximating the inner maximization with a single ascent step in the
    direction of the gradient.

    :param params: Model parameters.
    :param base_optimizer: Torch optimizer class (e.g., `torch.optim.SGD`).
    :param rho: Neighborhood size (perturbation radius).
    :param adaptive: If True, uses an adaptive perturbation (ASAM-style): scales
        the gradient by |w|.
    :param eps: Numerical stability epsilon for normalization.
    :param kwargs: Keyword arguments forwarded to `base_optimizer`.
    """

    def __init__(
        self,
        params,
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12,
        **kwargs: Any,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative, got: {rho}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps, should be positive, got: {eps}")

        defaults = dict(rho=rho, adaptive=adaptive, eps=eps, **kwargs)
        super().__init__(params, defaults)

        # The base optimizer will share the same param_groups object.
        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        # Make sure `param_groups` are those of the base optimizer (for serialization).
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute the L2 norm used to normalize the SAM perturbation."""
        norms: list[torch.Tensor] = []
        for group in self.param_groups:
            adaptive = bool(group.get("adaptive", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if adaptive:
                    grad = grad * p.abs()
                norms.append(torch.norm(grad, p=2))

        if len(norms) == 0:
            # Keep device/dtype consistent even if no grads.
            first_param = self.param_groups[0]["params"][0]
            return torch.zeros((), device=first_param.device, dtype=first_param.dtype)

        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, *, zero_grad: bool = True) -> None:
        """Move parameters to the local worst-case point `w + e(w)`.

        Call this after computing gradients at the current parameters.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            rho = float(group.get("rho", 0.05))
            eps = float(group.get("eps", 1e-12))
            adaptive = bool(group.get("adaptive", False))

            scale = rho / (grad_norm + eps)
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["old_p"] = p.detach().clone()

                e_w = p.grad
                if adaptive:
                    e_w = e_w * p.abs()
                e_w = e_w * scale
                p.add_(e_w)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, *, zero_grad: bool = True) -> None:
        """Restore parameters and apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old_p = self.state[p].get("old_p")
                if old_p is None:
                    raise RuntimeError("SAM.second_step() called before first_step()")
                p.copy_(old_p)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform a full SAM update.

        This requires a closure that reevaluates the model and returns the loss.
        The closure must perform a full forward + backward pass.
        """
        if closure is None:
            raise RuntimeError("SAM requires `closure` for `step()`")

        # First forward-backward pass
        loss = closure()
        self.first_step(zero_grad=True)

        # Second forward-backward pass at perturbed weights
        closure()
        self.second_step(zero_grad=True)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Delegate to base optimizer to match its behavior.
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        # Include base optimizer state for correctness.
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict)
        # Keep SAM param_groups in sync.
        self.param_groups = self.base_optimizer.param_groups
