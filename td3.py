import argparse

import torch

import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs import make_env
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchVecOnlineRLAlgorithm

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


def experiment(variant):
    dummy_env = make_env(variant['env'])
    obs_dim = dummy_env.observation_space.low.size
    action_dim = dummy_env.action_space.low.size
    expl_env = VectorEnv([lambda: make_env(variant['env']) for _ in range(variant['expl_env_num'])])
    expl_env.seed(variant["seed"])
    expl_env.action_space.seed(variant["seed"])
    eval_env = SubprocVectorEnv([lambda: make_env(variant['env']) for _ in range(variant['eval_env_num'])])
    eval_env.seed(variant["seed"])

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    es = GaussianStrategy(
        action_space=dummy_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )
    trainer = TD3Trainer(
        policy=policy,
        target_policy=target_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
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
    log_prefix = "_".join(["td3", variant["env"][:-3].lower(), str(variant["version"])])
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)
    set_seed(args.seed)
    experiment(variant)
