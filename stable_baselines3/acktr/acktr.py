from typing import Any, ClassVar, TypeVar

import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.acktr.kfac import KFACOptimizer
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

SelfACKTR = TypeVar("SelfACKTR", bound="ACKTR")


class ACKTR(OnPolicyAlgorithm):
    """
    Actor Critic using Kronecker-Factored Trust Region (ACKTR)

    Paper: https://arxiv.org/abs/1708.05144
    Code: This implementation is adapted from https://github.com/openai/baselines
    and https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

    Introduction to ACKTR: ACKTR is an actor-critic method that uses Kronecker-factored
    approximate curvature (K-FAC) for trust region optimization. It applies natural gradient
    descent using an approximation of the Fisher information matrix.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for K-FAC optimizer
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param kfac_update_freq: Frequency of updating K-FAC statistics (in number of optimizer steps)
    :param kfac_momentum: Momentum parameter for K-FAC
    :param kfac_damping: Damping parameter for K-FAC numerical stability
    :param kfac_kl_clip: KL divergence clip parameter for K-FAC
    :param kfac_stat_decay: Moving average decay for K-FAC statistics
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 0.25,
        n_steps: int = 20,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        vf_fisher_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        kfac_update_freq: int = 1,
        kfac_momentum: float = 0.9,
        kfac_damping: float = 1e-2,
        kfac_kl_clip: float = 0.001,
        kfac_stat_decay: float = 0.99,
        kfac_cold_start_steps: int = 10,
        kfac_cold_start_lr: float = 0.1,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        self.kfac_update_freq = kfac_update_freq
        self.kfac_momentum = kfac_momentum
        self.kfac_damping = kfac_damping
        self.kfac_kl_clip = kfac_kl_clip
        self.kfac_stat_decay = kfac_stat_decay
        self.kfac_cold_start_steps = kfac_cold_start_steps
        self.kfac_cold_start_lr = kfac_cold_start_lr
        self.vf_fisher_coef = vf_fisher_coef

        # K-FAC optimizer will be set up in _setup_model after policy is created
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Setup the model and create the K-FAC optimizer.
        """
        super()._setup_model()

        # Replace the optimizer with K-FAC optimizer
        # Get current learning rate (for scheduled learning rates, get initial value at progress 1.0)
        lr = self.learning_rate
        if callable(lr):
            # Start with the initial learning rate value
            lr = lr(1.0)

        self.policy.optimizer = KFACOptimizer(
            self.policy,
            lr=lr,
            momentum=self.kfac_momentum,
            stat_decay=self.kfac_stat_decay,
            kl_clip=self.kfac_kl_clip,
            damping=self.kfac_damping,
            update_freq=self.kfac_update_freq,
            weight_decay=0,
            cold_start_steps=self.kfac_cold_start_steps,
            cold_start_lr=self.kfac_cold_start_lr,
            max_grad_norm=self.max_grad_norm,
        )

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Only compute Fisher statistics when they will be consumed by the optimizer.
            # KFACOptimizer saves activations/gradients only when (steps % update_freq == 0)
            # and KFACOptimizer.step() consumes them at the beginning of the step.
            do_fisher_update = self.policy.optimizer.steps % self.policy.optimizer.update_freq == 0

            fisher_loss = None
            if do_fisher_update:
                # Fisher loss for K-FAC
                policy_fisher_loss = log_prob.mean()
                values_sample = (values + th.randn_like(values)).detach()
                value_fisher_loss = -((values - values_sample) ** 2).mean()
                fisher_loss = policy_fisher_loss + self.vf_fisher_coef * value_fisher_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward(retain_graph=do_fisher_update)

            if do_fisher_update:
                assert fisher_loss is not None
                # Use autograd.grad to run a backward pass that triggers hooks for Fisher
                # statistics without modifying parameter .grad (used for the actual update).
                self.policy.optimizer.acc_stats = True
                th.autograd.grad(
                    fisher_loss,
                    list(self.policy.parameters()),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                self.policy.optimizer.acc_stats = False

            # Step the K-FAC optimizer
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self: SelfACKTR,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "ACKTR",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfACKTR:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
