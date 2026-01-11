"""
Basic test for ACKTR implementation
"""
import gymnasium as gym
import pytest

from stable_baselines3 import ACKTR
from stable_baselines3.common.evaluation import evaluate_policy


def test_acktr_cartpole():
    """Test ACKTR on CartPole-v1"""
    env = gym.make("CartPole-v1")
    model = ACKTR("MlpPolicy", env, n_steps=64, verbose=0)
    
    # Train for a small number of timesteps
    model.learn(total_timesteps=500)
    
    # Test prediction
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    assert action is not None
    
    env.close()


def test_acktr_custom_hyperparams():
    """Test ACKTR with custom hyperparameters"""
    env = gym.make("CartPole-v1")
    model = ACKTR(
        "MlpPolicy",
        env,
        learning_rate=0.1,
        n_steps=32,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        kfac_update_freq=1,
        kfac_momentum=0.9,
        kfac_damping=1e-2,
        verbose=0,
    )
    
    model.learn(total_timesteps=500)
    
    env.close()


def test_acktr_save_load(tmp_path):
    """Test ACKTR save and load"""
    env = gym.make("CartPole-v1")
    model = ACKTR("MlpPolicy", env, n_steps=32, verbose=0)
    model.learn(total_timesteps=500)
    
    # Save model
    model_path = tmp_path / "acktr_cartpole"
    model.save(model_path)
    
    # Load model
    loaded_model = ACKTR.load(model_path, env=env)
    
    # Test prediction
    obs, _ = env.reset()
    action1, _ = model.predict(obs, deterministic=True)
    action2, _ = loaded_model.predict(obs, deterministic=True)
    
    assert action1 == action2
    
    env.close()


def test_acktr_policy_kwargs():
    """Test ACKTR with custom policy kwargs"""
    env = gym.make("CartPole-v1")
    model = ACKTR(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
        n_steps=32,
        verbose=0,
    )
    
    model.learn(total_timesteps=500)
    
    env.close()


if __name__ == "__main__":
    # Run basic tests
    test_acktr_cartpole()
    print("✓ test_acktr_cartpole passed")
    
    test_acktr_custom_hyperparams()
    print("✓ test_acktr_custom_hyperparams passed")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        from pathlib import Path
        test_acktr_save_load(Path(tmp_dir))
    print("✓ test_acktr_save_load passed")
    
    test_acktr_policy_kwargs()
    print("✓ test_acktr_policy_kwargs passed")
    
    print("\nAll tests passed!")
