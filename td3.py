import argparse

import gym
import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NoInfoEnv, NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm


def make_env(name):
    env = gym.make(name)
    env = NoInfoEnv(env)
    env = NormalizedBoxEnv(env)
    return env


def experiment(variant):
    expl_env = make_env(variant['env'])
    expl_env.seed(variant["seed"])
    eval_env = make_env(variant['env'])
    eval_env.seed(variant["seed"])
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpStepCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3Trainer(policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         target_qf1=target_qf1,
                         target_qf2=target_qf2,
                         target_policy=target_policy,
                         **variant['trainer_kwargs'])
    algorithm = TorchOnlineRLAlgorithm(trainer=trainer,
                                       exploration_env=expl_env,
                                       evaluation_env=eval_env,
                                       exploration_data_collector=expl_path_collector,
                                       evaluation_data_collector=eval_path_collector,
                                       replay_buffer=replay_buffer,
                                       **variant['algorithm_kwargs'])
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3')
    parser.add_argument('--config', type=str, default="configs/lunarlander.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = args.seed
    log_prefix = "_".join(["td3", variant["env"][:-3].lower(), variant["version"]])
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    set_seed(args.seed)
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)  # optionally set the GPU (default=False)
    experiment(variant)
