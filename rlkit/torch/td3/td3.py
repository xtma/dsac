from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class TD3Trainer(TorchTrainer):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            policy,
            target_policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=3e-4,
            qf_lr=3e-4,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,
            max_action=1.,
            clip_norm=0.,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy
        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.discount = discount
        self.reward_scale = reward_scale
        self.max_action = max_action
        self.clip_norm = clip_norm

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        """
        Update QF
        """
        with torch.no_grad():
            next_actions = self.target_policy(next_obs)
            noise = ptu.randn(next_actions.shape) * self.target_policy_noise
            noise = torch.clamp(noise, -self.target_policy_noise_clip, self.target_policy_noise_clip)
            noisy_next_actions = torch.clamp(next_actions + noise, -self.max_action, self.max_action)

            target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
            target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
            target_q_values = torch.min(target_q1_values, target_q2_values)
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target)**2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target)**2
        qf2_loss = bellman_errors_2.mean()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        """
        Update Policy
        """

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(obs)
            q_output = self.qf1(obs, policy_actions)
            policy_loss = -q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = -q_output.mean()

            self.eval_statistics['QF1 Loss'] = qf1_loss.item()
            self.eval_statistics['QF2 Loss'] = qf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.policy.state_dict(),
            target_policy=self.target_policy.state_dict(),
        )
