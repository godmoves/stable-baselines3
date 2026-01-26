"""ACKTR sanity benchmark on LunarLander-v3.

Goal: verify ACKTR + KFAC runs end-to-end on a harder continuous-control-ish task
(LunarLander is discrete actions) and produces a learning curve.

Default is intentionally *not* huge (so it finishes quickly). For a more reliable
conclusion about "solving" LunarLander, increase --total-timesteps (e.g. 200000+).

Run:
    python scripts/bench_acktr_lunarlander.py

Outputs:
- Prints evaluation rewards during training
- Saves a plot of mean eval reward vs timesteps
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running from a source checkout without installing the package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import ACKTR
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--log-dir", type=str, default=os.path.join(REPO_ROOT, "logs", "ACKTR_LunarLander"))
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(REPO_ROOT, "scripts", "acktr_lunarlander_eval.png"),
    )

    # ACKTR/KFAC hyperparams (defaults match the example, but can be overridden)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--vf-fisher-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    p.add_argument("--kfac-update-freq", type=int, default=1)
    p.add_argument("--kfac-stat-decay", type=float, default=0.95)
    p.add_argument("--kfac-damping", type=float, default=0.01)
    p.add_argument("--kfac-kl-clip", type=float, default=0.001)
    p.add_argument("--kfac-cold-start-steps", type=int, default=1000)
    p.add_argument("--kfac-cold-start-lr", type=float, default=0.001)
    p.add_argument("--normalize-advantage", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("=== ACKTR LunarLander benchmark ===")
    print(f"env={args.env_id}, seed={args.seed}, n_envs={args.n_envs}")
    print(f"total_timesteps={args.total_timesteps}, eval_freq={args.eval_freq}, n_eval_episodes={args.n_eval_episodes}")

    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed, monitor_dir=args.log_dir)
    eval_env = Monitor(make_vec_env(args.env_id, n_envs=1, seed=args.seed + 1))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.log_dir,
        log_path=args.log_dir,
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model = ACKTR(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        vf_fisher_coef=args.vf_fisher_coef,
        max_grad_norm=args.max_grad_norm,
        kfac_update_freq=args.kfac_update_freq,
        kfac_stat_decay=args.kfac_stat_decay,
        kfac_damping=args.kfac_damping,
        kfac_kl_clip=args.kfac_kl_clip,
        kfac_cold_start_steps=args.kfac_cold_start_steps,
        kfac_cold_start_lr=args.kfac_cold_start_lr,
        normalize_advantage=args.normalize_advantage,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    # Plot evaluation curve
    if eval_callback.evaluations_results is None or eval_callback.evaluations_timesteps is None:
        print("No evaluations collected (unexpected).")
        return

    timesteps = np.array(eval_callback.evaluations_timesteps)
    evals = np.array(eval_callback.evaluations_results)
    mean_rewards = evals.mean(axis=1)
    std_rewards = evals.std(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards, linewidth=2)
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Eval Reward")
    plt.title(f"ACKTR on {args.env_id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved plot to: {args.out}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
