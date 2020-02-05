from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from .networks import PopArt
from .risk import risk_fn
from .utils import LinearSchedule


def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()


class DSACTrainer(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            fp,
            target_fp,
            zf1,
            zf2,
            target_zf1,
            target_zf2,
            discount=0.99,
            reward_scale=1.0,
            entropy_penalty=0.01,
            policy_lr=3e-4,
            fp_lr=1e-4,
            zf_lr=3e-4,
            clip_grad=10000.,
            risk_type="neutral",
            risk_param=0.,
            risk_param_final=None,
            risk_schedule_timesteps=1,
            optimizer_class=optim.Adam,
            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            use_automatic_entropy_tuning=False,
            use_popart=False,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.fp = fp
        self.target_fp = target_fp
        self.zf1 = zf1
        self.zf2 = zf2
        self.target_zf1 = target_zf1
        self.target_zf2 = target_zf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_popart = use_popart
        if self.use_popart:
            self.zf1_normer = PopArt(self.zf1.last_fc)
            self.zf2_normer = PopArt(self.zf2.last_fc)
            self.target_zf1_normer = PopArt(self.target_zf1.last_fc)
            self.target_zf2_normer = PopArt(self.target_zf2.last_fc)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.zf_criterion = quantile_regression_loss
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.fp_optimizer = optimizer_class(
            self.fp.parameters(),
            lr=fp_lr,
        )
        self.zf1_optimizer = optimizer_class(
            self.zf1.parameters(),
            lr=zf_lr,
        )
        self.zf2_optimizer = optimizer_class(
            self.zf2.parameters(),
            lr=zf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.entropy_penalty = entropy_penalty
        self.clip_grad = clip_grad

        self.risk_type = risk_type
        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)

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
        Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 0.01

        presum_tau = self.fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        entropy_loss = -self.entropy_penalty * Categorical(presum_tau).entropy().mean()
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        """
        ZF Loss
        """
        # Make sure policy accounts for squashing functions like tanh correctly!
        with torch.no_grad():
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs,
                reparameterize=True,
                return_log_prob=True,
            )
            next_presum_tau = self.target_fp(next_obs, new_next_actions)
            next_tau = torch.cumsum(next_presum_tau, dim=1)  # (N, T)
            next_tau_hat = ptu.zeros_like(next_tau)
            next_tau_hat[:, 0:1] = next_tau[:, 0:1] / 2.
            next_tau_hat[:, 1:] = (next_tau[:, 1:] + next_tau[:, :-1]) / 2.

            target_z1_values = self.target_zf1(next_obs, new_next_actions, next_tau_hat)
            target_z2_values = self.target_zf2(next_obs, new_next_actions, next_tau_hat)
            if self.use_popart:
                target_z1_values = self.target_zf1_normer.unnorm(target_z1_values)
                target_z2_values = self.target_zf2_normer.unnorm(target_z2_values)
            target_q1_values = torch.sum(next_presum_tau * target_z1_values, dim=1, keepdims=True)
            target_q2_values = torch.sum(next_presum_tau * target_z2_values, dim=1, keepdims=True)
            target_q_values = torch.min(target_q1_values, target_q2_values)
            target_z1_values += -(target_q1_values - target_q_values) - alpha * new_log_pi
            target_z2_values += -(target_q2_values - target_q_values) - alpha * new_log_pi
            z1_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z1_values
            z2_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z2_values

            if self.use_popart:
                z1_target = self.zf1_normer.update(z1_target)
                z2_target = self.zf2_normer.update(z2_target)

        z1_pred = self.zf1(obs, actions, tau_hat)
        z2_pred = self.zf2(obs, actions, tau_hat)
        zf1_loss = self.zf_criterion(z1_pred, z1_target, tau_hat, next_presum_tau)
        zf2_loss = self.zf_criterion(z2_pred, z2_target, tau_hat, next_presum_tau)
        """
        Policy Loss
        """
        z1_new_actions = self.zf1(obs, new_obs_actions, tau_hat)
        z2_new_actions = self.zf2(obs, new_obs_actions, tau_hat)
        if self.use_popart:
            z1_new_actions = self.zf1_normer.unnorm(z1_new_actions)
            z2_new_actions = self.zf2_normer.unnorm(z2_new_actions)
        risk_param = self.risk_schedule(self._n_train_steps_total)
        with torch.no_grad():
            distorted_tau = risk_fn(tau, self.risk_type, risk_param)
            risk_weights = ptu.zeros_like(distorted_tau)
            risk_weights[:, 0:1] = distorted_tau[:, 0:1]
            risk_weights[:, 1:] = distorted_tau[:, 1:] - distorted_tau[:, :-1]
        q1_new_actions = torch.sum(risk_weights * z1_new_actions, dim=1, keepdims=True)
        q2_new_actions = torch.sum(risk_weights * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        if self.use_popart:
            policy_loss = 0.5 * (self.zf1_normer.norm(policy_loss) + self.zf2_normer.norm(policy_loss))
        """
        FP Grad
        """
        with torch.no_grad():
            dWdtau = 0.5 * sum([
                2 * self.zf1(obs, actions, tau[:, :-1]),
                -self.zf1(obs, actions, tau_hat[:, :-1]),
                -self.zf1(obs, actions, tau_hat[:, 1:]),
                2 * self.zf2(obs, actions, tau[:, :-1]),
                -self.zf2(obs, actions, tau_hat[:, :-1]),
                -self.zf2(obs, actions, tau_hat[:, 1:]),
            ])  # (N, T-1)
            dWdtau /= dWdtau.shape[0]
        """
        Update networks
        """
        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        zf1_grad = nn.utils.clip_grad_norm_(self.zf1.parameters(), self.clip_grad)
        self.zf1_optimizer.step()

        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        zf2_grad = nn.utils.clip_grad_norm_(self.zf2.parameters(), self.clip_grad)
        self.zf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.policy_optimizer.step()

        self.fp_optimizer.zero_grad()
        tau[:, :-1].backward(gradient=dWdtau, retain_graph=True)
        entropy_loss.backward()
        fp_grad = nn.utils.clip_grad_norm_(self.fp.parameters(), self.clip_grad)
        self.fp_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf1, self.target_zf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf2, self.target_zf2, self.soft_target_tau)
            if self.use_popart:
                ptu.soft_update_from_to(self.zf1_normer, self.target_zf1_normer, self.soft_target_tau)
                ptu.soft_update_from_to(self.zf2_normer, self.target_zf2_normer, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['ZF1 Loss'] = zf1_loss.item()
            self.eval_statistics['ZF2 Loss'] = zf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Entropy Loss'] = entropy_loss.item()
            self.eval_statistics['ZF1 Grad'] = zf1_grad
            self.eval_statistics['ZF2 Grad'] = zf2_grad
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics['FP Grad'] = fp_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z1 Predictions',
                ptu.get_numpy(z1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z2 Predictions',
                ptu.get_numpy(z2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z1 Targets',
                ptu.get_numpy(z1_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z2 Targets',
                ptu.get_numpy(z2_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        networks = [
            self.policy,
            self.fp,
            self.target_fp,
            self.zf1,
            self.zf2,
            self.target_zf1,
            self.target_zf2,
        ]
        if self.use_popart:
            networks += [
                self.zf1_normer,
                self.zf2_normer,
                self.target_zf1_normer,
                self.target_zf2_normer,
            ]
        return networks

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            fp=self.fp,
            target_fp=self.target_fp,
            zf1=self.zf1,
            zf2=self.zf2,
            target_zf1=self.zf1,
            target_zf2=self.zf2,
        )
        if self.use_popart:
            snapshot["zf1_normer"] = self.zf1_normer
            snapshot["zf2_normer"] = self.zf2_normer
            snapshot["target_zf1_normer"] = self.target_zf1_normer
            snapshot["target_zf2_normer"] = self.target_zf2_normer
        return snapshot
