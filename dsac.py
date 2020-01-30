import argparse

import torch

import gym
import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NoInfoEnv, NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.torch.dsac.dsac import DSACTrainer
from rlkit.torch.dsac.networks import QuantileMlp, softmax
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm
# import highway_env


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
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    num_quantiles = variant['num_quantiles']
    q_hidden_sizes = [M]
    z_hidden_sizes = [M, M, M // 4, M // 2]

    hidden_activation = variant['model_kwargs']['hidden_activation']
    if hidden_activation == "tanh":
        hidden_activation = torch.tanh
    elif hidden_activation == "relu":
        hidden_activation = torch.nn.functional.relu
    elif hidden_activation == "leaky_relu":
        hidden_activation = torch.nn.functional.leaky_relu
    else:
        raise NotImplementedError
    variant['model_kwargs']['hidden_activation'] = hidden_activation

    fp = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=num_quantiles,
        hidden_sizes=q_hidden_sizes,
        output_activation=softmax,
        **variant['model_kwargs'],
    )  # fraction proposal network
    target_fp = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=num_quantiles,
        hidden_sizes=q_hidden_sizes,
        output_activation=softmax,
        **variant['model_kwargs'],
    )  # fraction proposal network
    zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=z_hidden_sizes,
        **variant['model_kwargs'],
    )
    zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=z_hidden_sizes,
        **variant['model_kwargs'],
    )
    target_zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=z_hidden_sizes,
        **variant['model_kwargs'],
    )
    target_zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=z_hidden_sizes,
        **variant['model_kwargs'],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        **variant['model_kwargs'],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = DSACTrainer(
        env=eval_env,
        policy=policy,
        fp=fp,
        target_fp=target_fp,
        zf1=zf1,
        zf2=zf2,
        target_zf1=target_zf1,
        target_zf2=target_zf2,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchOnlineRLAlgorithm(
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
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Distributional Soft Actor Critic')
    parser.add_argument('--config', type=str, default="configs/lunarlander.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = args.seed
    log_prefix = "_".join(["dsac", variant["env"][:-3].lower(), variant["version"]])
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    set_seed(args.seed)
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)  # optionally set the GPU (default=False)
    experiment(variant)
