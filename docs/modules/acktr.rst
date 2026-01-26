.. _acktr:

.. automodule:: stable_baselines3.acktr


ACKTR
=====

`Actor Critic using Kronecker-Factored Trust Region (ACKTR) <https://arxiv.org/abs/1708.05144>`_ is an on-policy actor-critic algorithm that
uses Kronecker-factored approximate curvature (K-FAC) for natural gradient descent. It applies natural gradient
updates to both the actor and critic using an efficient approximation of the Fisher information matrix.

ACKTR improves upon A2C by using second-order optimization (natural gradients) which can lead to more stable
and faster training, especially for problems with difficult gradient landscapes.


Notes
-----

-  Original paper: https://arxiv.org/abs/1708.05144
-  OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
Dict          ❌     ✔️
============= ====== ===========


Example
-------

This example is only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized hyperparameters can be found in RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

Train an ACKTR agent on ``CartPole-v1`` using 4 environments.

.. code-block:: python

  from stable_baselines3 import ACKTR
  from stable_baselines3.common.env_util import make_vec_env

  # Parallel environments
  vec_env = make_vec_env("CartPole-v1", n_envs=4)

  model = ACKTR("MlpPolicy", vec_env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("acktr_cartpole")

  del model # remove to demonstrate saving and loading

  model = ACKTR.load("acktr_cartpole")

  obs = vec_env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render("human")


.. note::

  ACKTR is designed to run primarily on the CPU. The K-FAC optimizer requires computing Fisher information matrices
  which can be memory intensive on GPU. To improve CPU utilization, try using ``SubprocVecEnv`` instead of the default ``DummyVecEnv``:

  .. code-block:: python

    from stable_baselines3 import ACKTR
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if __name__=="__main__":
        env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
        model = ACKTR("MlpPolicy", env, device="cpu")
        model.learn(total_timesteps=25_000)

  For more information, see :ref:`Vectorized Environments <vec_env>`.


Parameters
----------

.. autoclass:: ACKTR
  :members:
  :inherited-members:


.. _acktr_policies:

ACKTR Policies
--------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.common.policies.ActorCriticPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: stable_baselines3.common.policies.ActorCriticCnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputPolicy
  :members:

.. autoclass:: stable_baselines3.common.policies.MultiInputActorCriticPolicy
  :members:
  :noindex:
