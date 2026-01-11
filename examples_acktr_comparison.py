"""
Algorithm Comparison: ACKTR vs PPO vs A2C

This script compares ACKTR, PPO, and A2C algorithms on LunarLander-v2,
a more challenging environment than CartPole. It trains each algorithm
and plots their learning curves for comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import A2C, ACKTR, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Environment settings
ENV_ID = "LunarLander-v2"
N_ENVS = 4
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10

print(f"Comparing ACKTR, PPO, and A2C on {ENV_ID}")
print(f"Training for {TOTAL_TIMESTEPS} timesteps\n")

# Results storage
results = {}

# Algorithm configurations
algorithms = {
    "ACKTR": {
        "class": ACKTR,
        "kwargs": {
            "learning_rate": 0.01,  # Reduced from 0.25 - too high for LunarLander
            "n_steps": 128,  # Increased from 20 - larger batch size for stability
            "gamma": 0.99,
            "gae_lambda": 0.95,  # Changed from 1.0 - better bias-variance tradeoff
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "kfac_update_freq": 10,  # Update K-FAC less frequently for stability
            "kfac_damping": 0.1,  # Increased damping for stability
            "normalize_advantage": True,  # Enable advantage normalization
        },
    },
    "PPO": {
        "class": PPO,
        "kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
    },
    "A2C": {
        "class": A2C,
        "kwargs": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
    },
}

# Train and evaluate each algorithm
for algo_name, algo_config in algorithms.items():
    print(f"\n{'='*50}")
    print(f"Training {algo_name}")
    print(f"{'='*50}")

    # Create vectorized environment
    env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    eval_env = make_vec_env(ENV_ID, n_envs=1)

    # Create callback for evaluation during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{algo_name}/",
        log_path=f"./logs/{algo_name}/",
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # Create and train model
    model = algo_config["class"](
        "MlpPolicy", env, verbose=0, **algo_config["kwargs"]
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    # Final evaluation
    print(f"\nFinal evaluation of {algo_name}...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    print(f"{algo_name} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Store results
    results[algo_name] = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "evaluations": eval_callback.evaluations_results,
        "timesteps": eval_callback.evaluations_timesteps,
    }

    # Clean up
    env.close()
    eval_env.close()

# Plot comparison
print("\n" + "=" * 50)
print("Generating comparison plot...")
print("=" * 50)

plt.figure(figsize=(12, 6))

for algo_name, result in results.items():
    timesteps = np.array(result["timesteps"])
    evaluations = np.array(result["evaluations"])
    mean_rewards = evaluations.mean(axis=1)
    std_rewards = evaluations.std(axis=1)

    plt.plot(timesteps, mean_rewards, label=algo_name, linewidth=2)
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
    )

plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Mean Reward", fontsize=12)
plt.title(f"Algorithm Comparison on {ENV_ID}", fontsize=14, fontweight="bold")
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plot_filename = "algorithm_comparison.png"
plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"\nPlot saved as '{plot_filename}'")

# Print final summary
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
for algo_name, result in results.items():
    print(
        f"{algo_name:10s}: {result['mean_reward']:7.2f} +/- {result['std_reward']:5.2f}"
    )

print("\nComparison complete!")
