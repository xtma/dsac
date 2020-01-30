import argparse

import gym
import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NoInfoEnv, NormalizedBoxEnv, HighwayWrapper
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm


def make_env(name):
    env = gym.make(name)
    if name.startswith("summon"):
        env = HighwayWrapper(env)
    else:
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
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
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
    trainer = SACTrainer(env=eval_env,
                         policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         target_qf1=target_qf1,
                         target_qf2=target_qf2,
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
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Soft Actor Critic')
    parser.add_argument('--config', type=str, default="configs/lunarlander.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = args.seed
    log_prefix = "_".join(["sac", variant["env"][:-3].lower(), variant["version"]])
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    set_seed(args.seed)
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)  # optionally set the GPU (default=False)
    experiment(variant)
