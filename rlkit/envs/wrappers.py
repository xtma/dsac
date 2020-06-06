import numpy as np
import itertools
from gym import Env, Wrapper
from gym.spaces import Box
from gym.spaces import Discrete
from collections import deque


class ProxyEnv(Env):

    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class HistoryEnv(ProxyEnv, Env):

    def __init__(self, wrapped_env, history_len):
        super().__init__(wrapped_env)
        self.history_len = history_len

        high = np.inf * np.ones(self.history_len * self.observation_space.low.size)
        low = -high
        self.observation_space = Box(
            low=low,
            high=high,
        )
        self.history = deque(maxlen=self.history_len)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history, reward, done, info

    def reset(self, **kwargs):
        state = super().reset()
        self.history = deque(maxlen=self.history_len)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_len - obs_count):
            dummy = np.zeros(self._wrapped_env.observation_space.low.size)
            observations.append(dummy)
        return np.c_[observations]


class DiscretizeEnv(ProxyEnv, Env):

    def __init__(self, wrapped_env, num_bins):
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [np.linspace(low[i], high[i], num_bins) for i in range(len(low))]
        self.idx_to_continuous_action = [np.array(x) for x in itertools.product(*action_ranges)]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)


class NormalizedBoxEnv(Wrapper):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        super().__init__(env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self.env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To " "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)
        self._should_normalize = True

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self.env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self.env


class CustomInfoEnv(Wrapper):

    def __init__(self, wrapped_env):

        env_id = wrapped_env.spec.id
        if env_id in [
                "Walker2d-v2",  # mujoco
                "Hopper-v2",
                "Ant-v2",
                "HalfCheetah-v2",
                "Humanoid-v2",
                "HumanoidStandup-v2",
                "Walker2d-v3",  # mujoco
                "Hopper-v3",
                "Ant-v3",
                "HalfCheetah-v3",
                "Humanoid-v3",
        ]:
            self.env_type = "mujoco"
        elif env_id in [
                "LunarLanderContinuous-v2",
                "BipedalWalker-v3",
                "BipedalWalkerHardcore-v3",
        ]:
            self.env_type = "box2d"

        super().__init__(wrapped_env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.env_type == "mujoco":
            custom_info = {'failed': done}
        if self.env_type == "box2d":
            custom_info = {'failed': reward <= -100}
        return state, reward, done, custom_info
