import gym
from gym.wrappers import TimeLimit

from rlkit.envs.wrappers import CustomInfoEnv, NormalizedBoxEnv


def make_env(name):
    env = gym.make(name)
    # Remove TimeLimit Wrapper
    if isinstance(env, TimeLimit):
        env = env.unwrapped
    env = CustomInfoEnv(env)
    env = NormalizedBoxEnv(env)
    return env
