"""Quick optimizer comparison on Fashion-MNIST (1 epoch by default).

Compares SGD, Adam, and SB3's KFACOptimizer on the same small CNN.
Outputs a single figure (loss + accuracy subplots) for a quick sanity check
of *practical* optimization behavior.

Run (defaults are chosen to be quick):
    python scripts/bench_kfac_fashion_mnist.py

Notes about KFAC in supervised learning:
- We compute the supervised gradient from true labels.
- We update KFAC Fisher statistics using a separate backward pass where labels
  are sampled from the model distribution (as commonly done in KFAC/ACKTR-style
  implementations).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Allow running from a source checkout without installing the package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3.acktr.kfac import KFACOptimizer
from stable_baselines3.acktr.gnc import GNCOptimizer
from stable_baselines3.sam import SAM
from stable_baselines3.common.block_shampoo import BlockShampoo

try:
    from torch_optimizer import Shampoo as ShampooOptimizer
except ImportError:  # pragma: no cover
    ShampooOptimizer = None


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class Curve:
    steps: list[int]
    loss: list[float]
    acc: list[float]
    step_time_ms: list[float]
    grad_cos: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def make_dataloaders(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for Fashion-MNIST. Install it with: pip install torchvision"
        ) from e

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])

    train_ds = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(seed)

    if train_samples < len(train_ds):
        idx = torch.randperm(len(train_ds), generator=g)[:train_samples].tolist()
        train_ds = Subset(train_ds, idx)

    if test_samples < len(test_ds):
        idx = torch.randperm(len(test_ds), generator=g)[:test_samples].tolist()
        test_ds = Subset(test_ds, idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, test_loader


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return total_loss / max(1, total), total_correct / max(1, total)


def kfac_train_step(
    model: nn.Module,
    opt: KFACOptimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    """One supervised step using KFAC.

    - supervised backward for gradients
    - separate Fisher-stat backward with sampled labels (only when opt.steps % update_freq == 0)
    - optimizer step uses the supervised gradients
    """

    model.train()

    # Ensure inputs require grad to avoid full-backward-hook warnings in some PyTorch versions.
    x = x.detach().requires_grad_(True)

    # 1) Supervised gradient
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="mean")
    loss.backward()
    acc = accuracy_from_logits(logits.detach(), y)

    # Save supervised grads
    saved_grads: list[torch.Tensor | None] = []
    for p in model.parameters():
        saved_grads.append(None if p.grad is None else p.grad.detach().clone())

    # 2) Fisher stats update
    if opt.steps % opt.update_freq == 0:
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits_ng = model(x)
            probs = torch.softmax(logits_ng, dim=1)
            y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)

        opt.acc_stats = True
        logits_f = model(x)
        fisher_loss = F.cross_entropy(logits_f, y_sample, reduction="mean")
        fisher_loss.backward()
        opt.acc_stats = False

        # Note: we intentionally do NOT call opt._update_fisher_stats() here.
        # KFACOptimizer.step() will consume the saved activations/gradients and update statistics
        # at the beginning of the step when opt.steps % opt.update_freq == 0.

    # 3) Restore supervised grads and take step
    for p, g in zip(model.parameters(), saved_grads):
        p.grad = g

    opt.step()
    return float(loss.item()), acc


def gnc_train_step(
    model: nn.Module,
    opt: GNCOptimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    """One supervised step using GNC.

    GNC collects activation/gradient statistics via module hooks.
    This implementation relies on the optimizer's internal `update_freq` gating.
    """

    model.train()

    # Ensure inputs require grad to avoid full-backward-hook warnings in some PyTorch versions.
    x = x.detach().requires_grad_(True)

    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="mean")
    loss.backward()

    acc = accuracy_from_logits(logits.detach(), y)
    opt.step()
    return float(loss.item()), acc


def sam_train_step(
    model: nn.Module,
    opt: SAM,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float, float]:
    """One supervised step using SAM.

    Records:
    - loss/acc from the first (unperturbed) forward/backward
    - cosine similarity between first and second gradients (w vs w + eps)
    """

    model.train()

    def closure_record_metrics() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()
        return loss, logits, y

    # 1) First gradient at current parameters
    loss, logits, y_ref = closure_record_metrics()
    loss_value = float(loss.item())
    acc_value = accuracy_from_logits(logits.detach(), y_ref)

    grads_0: list[torch.Tensor | None] = []
    for p in model.parameters():
        grads_0.append(None if p.grad is None else p.grad.detach().clone())

    # 2) Move to the SAM neighborhood and compute second gradient
    opt.first_step(zero_grad=True)
    _loss_2, _logits_2, _y_2 = closure_record_metrics()

    dot = torch.zeros((), device=x.device)
    n0 = torch.zeros((), device=x.device)
    n1 = torch.zeros((), device=x.device)
    for g0, p in zip(grads_0, model.parameters()):
        g1 = p.grad
        if g0 is None or g1 is None:
            continue
        dot = dot + (g0 * g1).sum()
        n0 = n0 + (g0 * g0).sum()
        n1 = n1 + (g1 * g1).sum()

    denom = torch.sqrt(n0) * torch.sqrt(n1)
    if float(denom.item()) == 0.0:
        cos_sim = float("nan")
    else:
        cos_sim = float((dot / denom).clamp(-1.0, 1.0).item())

    # 3) Restore parameters and take the base optimizer step using the second gradient
    opt.second_step(zero_grad=True)
    return loss_value, acc_value, cos_sim


def make_optimizer(kind: str, model: nn.Module, args: argparse.Namespace):
    kind = kind.lower()
    if kind == "sgd":
        use_nesterov = bool(args.nesterov) and float(args.momentum) > 0.0
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr_sgd,
            momentum=args.momentum,
            nesterov=use_nesterov,
            weight_decay=args.weight_decay,
        )
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr_adam, weight_decay=args.weight_decay)
    if kind == "shampoo":
        if ShampooOptimizer is None:
            raise RuntimeError(
                "Shampoo optimizer requires 'torch-optimizer'. Install it with: pip install torch-optimizer"
            )
        return ShampooOptimizer(
            model.parameters(),
            lr=args.lr_shampoo,
            momentum=args.shampoo_momentum,
            weight_decay=args.weight_decay,
            epsilon=args.shampoo_epsilon,
            update_freq=args.shampoo_update_freq,
        )
    if kind in {"block_shampoo", "blockshampoo"}:
        return BlockShampoo(
            model.parameters(),
            lr=args.lr_block_shampoo,
            momentum=args.block_shampoo_momentum,
            weight_decay=args.weight_decay,
            epsilon=args.block_shampoo_epsilon,
            update_freq=args.block_shampoo_update_freq,
            block_size=args.block_shampoo_block_size,
            stat_decay=args.block_shampoo_stat_decay,
        )
    if kind == "sam":
        # Use SGD as the base optimizer for SAM (common default).
        use_nesterov = bool(args.nesterov) and float(args.momentum) > 0.0
        return SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=args.lr_sgd,
            momentum=args.momentum,
            nesterov=use_nesterov,
            weight_decay=args.weight_decay,
            rho=0.05,
        )
    if kind == "kfac":
        return KFACOptimizer(
            model,
            lr=args.lr_kfac,
            momentum=args.momentum,
            stat_decay=args.stat_decay,
            damping=args.damping,
            kl_clip=args.kl_clip,
            weight_decay=args.weight_decay,
            update_freq=args.update_freq,
            cold_start_steps=args.cold_start_steps,
            max_grad_norm=args.max_grad_norm,
        )
    if kind == "gnc":
        # GNC often needs separate learning rates for stability.
        return GNCOptimizer(
            model,
            lr=args.lr_gnc,
            momentum=args.gnc_momentum,
            factor_lr=args.factor_lr_gnc,
            factor_momentum=args.gnc_factor_momentum,
            damping=args.gnc_damping,
            stat_decay=args.gnc_stat_decay,
            weight_decay=args.weight_decay,
            update_freq=args.gnc_update_freq,
            cold_start_steps=args.gnc_cold_start_steps,
            max_grad_norm=args.gnc_max_grad_norm,
        )
    raise ValueError(f"Unknown optimizer: {kind}")


def train_one_epoch(
    *,
    model: nn.Module,
    optimizer,
    optimizer_kind: str,
    train_loader: DataLoader,
    device: torch.device,
    max_steps: int | None,
) -> Curve:
    curve = Curve(steps=[], loss=[], acc=[], step_time_ms=[], grad_cos=[])
    model.train()

    global_step = 0
    for x, y in train_loader:
        if max_steps is not None and global_step >= max_steps:
            break

        x = x.to(device)
        y = y.to(device)

        t0 = time.perf_counter()
        if optimizer_kind == "kfac":
            loss_value, acc_value = kfac_train_step(model, optimizer, x, y)
            cos_value = float("nan")
        elif optimizer_kind == "gnc":
            loss_value, acc_value = gnc_train_step(model, optimizer, x, y)
            cos_value = float("nan")
        elif optimizer_kind == "sam":
            loss_value, acc_value, cos_value = sam_train_step(model, optimizer, x, y)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            acc_value = accuracy_from_logits(logits.detach(), y)
            cos_value = float("nan")
        t1 = time.perf_counter()

        curve.steps.append(global_step)
        curve.loss.append(loss_value)
        curve.acc.append(acc_value)
        curve.step_time_ms.append((t1 - t0) * 1000.0)
        curve.grad_cos.append(cos_value)

        global_step += 1

    return curve


def plot_curves(curves: dict[str, Curve], out_path: str, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, c in curves.items():
        axes[0].plot(c.steps, c.loss, label=name)
    axes[0].set_ylabel("train loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for name, c in curves.items():
        axes[1].plot(c.steps, c.acc, label=name)
    axes[1].set_ylabel("train acc")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sam_grad_cos(curve: Curve, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(curve.steps, curve.grad_cos, label="cos(g0, g1)")
    ax.set_xlabel("step")
    ax.set_ylabel("grad cosine")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--data-dir", type=str, default=os.path.join(REPO_ROOT, "data"))

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None, help="Optional max steps per epoch for faster runs")

    p.add_argument(
        "--optimizers",
        nargs="+",
        default=["sgd", "adam", "shampoo", "block_shampoo", "sam", "kfac", "gnc"],
        choices=["sgd", "adam", "shampoo", "block_shampoo", "sam", "kfac", "gnc"],
        help="Which optimizers to run (default: all). Useful for quick sweeps.",
    )

    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--test-samples", type=int, default=2000)

    # Baselines
    p.add_argument("--lr-sgd", type=float, default=0.05)
    p.add_argument("--lr-adam", type=float, default=1e-2)
    p.add_argument("--lr-shampoo", type=float, default=0.2)
    p.add_argument("--shampoo-momentum", type=float, default=0.0)
    p.add_argument("--shampoo-epsilon", type=float, default=1e-3)
    p.add_argument("--shampoo-update-freq", type=int, default=1)

    p.add_argument("--lr-block-shampoo", type=float, default=0.01)
    p.add_argument("--block-shampoo-momentum", type=float, default=0.0)
    p.add_argument("--block-shampoo-epsilon", type=float, default=1e-4)
    p.add_argument("--block-shampoo-update-freq", type=int, default=10)
    p.add_argument("--block-shampoo-block-size", type=int, default=256)
    p.add_argument("--block-shampoo-stat-decay", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Nesterov momentum for SGD/SAM base optimizer (only when momentum > 0).",
    )

    # KFAC
    p.add_argument("--lr-kfac", type=float, default=0.25)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--stat-decay", type=float, default=0.95)
    p.add_argument("--damping", type=float, default=1e-2)
    p.add_argument("--kl-clip", type=float, default=1e-2)
    p.add_argument("--update-freq", type=int, default=10)
    p.add_argument("--cold-start-steps", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=None)

    # GNC (separate knobs from KFAC so we can tune independently)
    p.add_argument("--lr-gnc", type=float, default=0.01)
    p.add_argument("--factor-lr-gnc", type=float, default=0.05)
    p.add_argument("--gnc-momentum", type=float, default=0.9)
    p.add_argument("--gnc-factor-momentum", type=float, default=0.9)
    p.add_argument("--gnc-damping", type=float, default=0.3)
    p.add_argument("--gnc-stat-decay", type=float, default=0.95)
    p.add_argument("--gnc-update-freq", type=int, default=10)
    p.add_argument("--gnc-cold-start-steps", type=int, default=100)
    p.add_argument("--gnc-max-grad-norm", type=float, default=0.5)

    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(REPO_ROOT, "scripts", "bench_fashion_mnist_optimizers.png"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    train_loader, test_loader = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )

    optimizers = list(args.optimizers)
    curves: dict[str, Curve] = {}

    print("=== Fashion-MNIST quick benchmark ===")
    print(f"device={device}, seed={args.seed}, epochs={args.epochs}, batch={args.batch_size}, train_samples={args.train_samples}")

    for opt_name in optimizers:
        set_seed(args.seed)
        model = SmallCNN(num_classes=10).to(device)
        optimizer = make_optimizer(opt_name, model, args)

        all_steps: list[int] = []
        all_loss: list[float] = []
        all_acc: list[float] = []
        all_time: list[float] = []
        all_cos: list[float] = []

        t0 = time.perf_counter()
        for _ in range(args.epochs):
            curve = train_one_epoch(
                model=model,
                optimizer=optimizer,
                optimizer_kind=opt_name,
                train_loader=train_loader,
                device=device,
                max_steps=args.max_steps,
            )
            # Stitch epochs into one curve
            offset = len(all_steps)
            all_steps.extend([s + offset for s in curve.steps])
            all_loss.extend(curve.loss)
            all_acc.extend(curve.acc)
            all_time.extend(curve.step_time_ms)
            all_cos.extend(curve.grad_cos)
        t1 = time.perf_counter()

        test_loss, test_acc = eval_model(model, test_loader, device)
        mean_step_ms = float(np.mean(all_time)) if all_time else float("nan")

        curves[opt_name.upper()] = Curve(
            steps=all_steps,
            loss=all_loss,
            acc=all_acc,
            step_time_ms=all_time,
            grad_cos=all_cos,
        )
        print(
            f"{opt_name.upper():4s}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  mean_step_ms={mean_step_ms:.2f}  wall_s={(t1 - t0):.1f}"
        )

    title = "Fashion-MNIST (1 epoch) - SGD vs Adam vs SAM vs KFAC"
    plot_curves(curves, out_path=args.out, title=title)
    print(f"Saved plot to: {args.out}")

    sam_key = "SAM"
    if sam_key in curves:
        cos_out = os.path.splitext(args.out)[0] + "_sam_grad_cos.npy"
        np.save(cos_out, np.asarray(curves[sam_key].grad_cos, dtype=np.float32))
        print(f"Saved SAM grad cosine curve to: {cos_out}")

        cos_png = os.path.splitext(args.out)[0] + "_sam_grad_cos.png"
        plot_sam_grad_cos(
            curves[sam_key],
            out_path=cos_png,
            title="Fashion-MNIST - SAM grad cosine (first vs second)"
        )
        print(f"Saved SAM grad cosine plot to: {cos_png}")


if __name__ == "__main__":
    main()
