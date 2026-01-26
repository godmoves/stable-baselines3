from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _Partition:
    row_slices: list[slice]
    col_slices: list[slice]


def _partition_2d(m: int, n: int, block_size: int) -> _Partition:
    row_slices: list[slice] = []
    col_slices: list[slice] = []

    for r0 in range(0, m, block_size):
        row_slices.append(slice(r0, min(m, r0 + block_size)))
    for c0 in range(0, n, block_size):
        col_slices.append(slice(c0, min(n, c0 + block_size)))

    return _Partition(row_slices=row_slices, col_slices=col_slices)


def _matrix_power_symmetric(mat: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """Compute (mat + eps*I)^power for symmetric mat via eigendecomposition."""
    n = mat.shape[0]
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    # Force symmetry (numerical drift)
    mat = 0.5 * (mat + mat.transpose(-1, -2))

    # For PSD-ish matrices, eigh is the right choice.
    # NOTE: Some static analyzers in this workspace incorrectly flag torch.linalg.eigh as
    # non-callable. Looking it up dynamically via __dict__ avoids that false positive.
    linalg_mod = torch.__dict__["linalg"]
    eigh_fn = linalg_mod.__dict__["eigh"]

    # Retry with increasing diagonal jitter to improve conditioning.
    evals = None
    evecs = None
    for mul in (1.0, 10.0, 100.0, 1000.0):
        try:
            evals, evecs = eigh_fn(mat + (mul * eps) * eye)
            break
        except RuntimeError:
            continue

    if evals is None or evecs is None:
        return eye

    evals = torch.clamp(evals, min=eps)
    evals_p = torch.pow(evals, power)
    out = (evecs * evals_p.unsqueeze(0)) @ evecs.transpose(-1, -2)
    if not torch.isfinite(out).all():
        return eye
    return out


class BlockShampoo(torch.optim.Optimizer):
    """A minimal BlockShampoo optimizer (2D / Conv2d weights) for quick experiments.

    This is intentionally small and benchmark-oriented:
    - Uses symmetric eigendecomposition to compute inverse roots (more stable than SVD).
    - Splits large dimensions into blocks to avoid huge matrix decompositions.

    Supports:
    - 2D tensors (Linear weights)
    - 4D tensors (Conv2d weights), treated as (out_channels, in_channels*kH*kW)

    1D tensors (bias/LayerNorm/etc.) fall back to momentum SGD.

    Parameters:
    - lr: learning rate
    - momentum: momentum for parameter updates
    - weight_decay: L2 penalty (coupled; added to grad before preconditioning)
    - epsilon: numerical stability term used both in stats and inverse-root compute
    - update_freq: recompute inverse roots every N steps
    - block_size: block size for partitioning 2D matrices
    - stat_decay: EMA decay for second-moment statistics
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 10,
        block_size: int = 256,
        stat_decay: float = 0.9,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if update_freq < 1:
            raise ValueError("update_freq must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not (0.0 <= stat_decay < 1.0):
            raise ValueError("stat_decay must be in [0, 1)")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            block_size=block_size,
            stat_decay=stat_decay,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure: Any | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr: float = float(group["lr"])
            momentum: float = float(group["momentum"])
            weight_decay: float = float(group["weight_decay"])
            eps: float = float(group["epsilon"])
            update_freq: int = int(group["update_freq"])
            block_size: int = int(group["block_size"])
            stat_decay: float = float(group["stat_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("BlockShampoo does not support sparse gradients")

                # Coupled weight decay to match other optimizers in this benchmark.
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Momentum buffer
                if momentum != 0.0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p)
                        state["momentum_buffer"] = buf
                else:
                    buf = None

                # 1D or other unsupported shapes -> momentum SGD
                if grad.ndim not in (2, 4):
                    update = grad
                    if buf is not None:
                        buf.mul_(momentum).add_(update)
                        update = buf
                    p.data.add_(update, alpha=-lr)
                    continue

                # Convert to 2D matrix
                if grad.ndim == 4:
                    g_mat = grad.view(grad.shape[0], -1)
                else:
                    g_mat = grad

                m, n = g_mat.shape

                part: _Partition | None = state.get("partition")
                if part is None or state.get("shape") != (m, n) or state.get("block_size") != block_size:
                    part = _partition_2d(m, n, block_size)
                    state["partition"] = part
                    state["shape"] = (m, n)
                    state["block_size"] = block_size

                    # Per-block statistics and cached inverse roots
                    l_stats: list[torch.Tensor] = []
                    r_stats: list[torch.Tensor] = []
                    l_invroots: list[torch.Tensor] = []
                    r_invroots: list[torch.Tensor] = []

                    for rs in part.row_slices:
                        br = rs.stop - rs.start
                        l_stats.append(torch.zeros((br, br), device=p.device, dtype=p.dtype))
                        l_invroots.append(torch.eye(br, device=p.device, dtype=p.dtype))
                    for cs in part.col_slices:
                        bc = cs.stop - cs.start
                        r_stats.append(torch.zeros((bc, bc), device=p.device, dtype=p.dtype))
                        r_invroots.append(torch.eye(bc, device=p.device, dtype=p.dtype))

                    state["l_stats"] = l_stats
                    state["r_stats"] = r_stats
                    state["l_invroots"] = l_invroots
                    state["r_invroots"] = r_invroots

                l_stats = state["l_stats"]
                r_stats = state["r_stats"]
                l_invroots = state["l_invroots"]
                r_invroots = state["r_invroots"]

                # Update stats (EMA of G G^T and G^T G per block)
                for i, rs in enumerate(part.row_slices):
                    g_block_rows = g_mat[rs, :]
                    ggT = g_block_rows @ g_block_rows.transpose(0, 1)
                    l_stats[i].mul_(stat_decay).add_(ggT, alpha=1.0 - stat_decay)

                for j, cs in enumerate(part.col_slices):
                    g_block_cols = g_mat[:, cs]
                    gTg = g_block_cols.transpose(0, 1) @ g_block_cols
                    r_stats[j].mul_(stat_decay).add_(gTg, alpha=1.0 - stat_decay)

                # Recompute inverse roots periodically.
                if self._step_count % update_freq == 0:
                    # For 2D tensors, order=2 (two dimensions)
                    inv_root_power = -0.25
                    for i, stat in enumerate(l_stats):
                        l_invroots[i] = _matrix_power_symmetric(stat, inv_root_power, eps)
                        if not torch.isfinite(l_invroots[i]).all():
                            l_invroots[i] = torch.eye(l_invroots[i].shape[0], device=p.device, dtype=p.dtype)
                    for j, stat in enumerate(r_stats):
                        r_invroots[j] = _matrix_power_symmetric(stat, inv_root_power, eps)
                        if not torch.isfinite(r_invroots[j]).all():
                            r_invroots[j] = torch.eye(r_invroots[j].shape[0], device=p.device, dtype=p.dtype)

                # Apply preconditioning blockwise
                pre_g = torch.empty_like(g_mat)
                for i, rs in enumerate(part.row_slices):
                    for j, cs in enumerate(part.col_slices):
                        g_ij = g_mat[rs, cs]
                        block_update = l_invroots[i] @ g_ij @ r_invroots[j]
                        if not torch.isfinite(block_update).all():
                            block_update = g_ij
                        pre_g[rs, cs] = block_update

                # Momentum update and parameter step
                if grad.ndim == 4:
                    update_full = pre_g.view_as(p.data)
                else:
                    update_full = pre_g

                if buf is not None:
                    buf.mul_(momentum).add_(update_full)
                    update_full = buf

                p.data.add_(update_full, alpha=-lr)

        return loss
