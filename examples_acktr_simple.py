"""
Simple example demonstrating ACKTR usage

This script shows how to train and use an ACKTR agent on CartPole-v1
"""

import gymnasium as gym

from stable_baselines3 import ACKTR
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env("CartPole-v1", n_envs=4)

# Create ACKTR model with default hyperparameters
print("Creating ACKTR model...")
model = ACKTR("MlpPolicy", env, verbose=1)

# Train the agent
print("\nTraining for 25000 timesteps...")
model.learn(total_timesteps=25000)

# Save the model
print("\nSaving model...")
model.save("acktr_cartpole")

# Load the model
print("Loading model...")
loaded_model = ACKTR.load("acktr_cartpole")

# Test the trained agent
print("\nTesting trained agent...")
test_env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, _ = test_env.reset()
total_reward = 0
for _ in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode finished with total reward: {total_reward}")
        total_reward = 0
        obs, _ = test_env.reset()

test_env.close()
env.close()

print("\nDone!")
