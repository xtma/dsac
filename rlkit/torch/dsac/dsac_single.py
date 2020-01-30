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


def quantile_regression_loss(input, target, tau):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    # L = F.mse_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L
    return rho.mean()


class SingleDSACTrainer(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            fp,
            target_fp,
            zf,
            target_zf,
            discount=0.99,
            reward_scale=1.0,
            entropy_penalty=0.01,
            policy_lr=1e-3,
            fp_lr=1e-3,
            zf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            use_automatic_entropy_tuning=True,
            use_popart=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.fp = fp
        self.target_fp = target_fp
        self.zf = zf
        self.target_zf = target_zf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_popart = use_popart
        if self.use_popart:
            self.zf_normer = PopArt(self.zf.last_fc)
            self.target_zf_normer = PopArt(self.target_zf.last_fc)

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
        self.zf_optimizer = optimizer_class(
            self.zf.parameters(),
            lr=zf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.entropy_penalty = entropy_penalty
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
        entropy_loss = -self.entropy_penalty * Categorical(presum_tau).entropy().mean()
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = tau / 2.
            tau_hat[:, 1:] += tau_hat[:, :-1]
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
            next_tau_hat = next_tau / 2.
            next_tau_hat[:, 1:] += next_tau_hat[:, :-1]
            target_z_values = self.target_zf(next_obs, new_next_actions, next_tau_hat)
            if self.use_popart:
                target_z_values = self.target_zf_normer.unnorm(target_z_values)
            target_z_values += -alpha * new_log_pi
            z_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z_values
            if self.use_popart:
                z_target = self.zf_normer.update(z_target)

        z_pred = self.zf(obs, actions, tau_hat)
        zf_loss = self.zf_criterion(z_pred, z_target, tau_hat)
        """
        Policy Loss
        """
        z_new_actions = self.zf(obs, new_obs_actions, tau_hat)
        if self.use_popart:
            z_new_actions = self.zf_normer.unnorm(z_new_actions)
        q_new_actions = torch.sum(presum_tau.detach() * z_new_actions, dim=1, keepdims=True)
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        if self.use_popart:
            policy_loss = self.zf_normer.norm(policy_loss)
        """
        FP Grad
        """
        with torch.no_grad():
            dWdtau = sum([
                2 * self.zf(obs, actions, tau[:, :-1]),
                -self.zf(obs, actions, tau_hat[:, :-1]),
                -self.zf(obs, actions, tau_hat[:, 1:]),
            ])  # (N, T-1)
            dWdtau /= dWdtau.shape[0]
        """
        Update networks
        """
        self.zf_optimizer.zero_grad()
        zf_loss.backward()
        zf_grad = nn.utils.clip_grad_norm_(self.zf.parameters(), 1.)
        self.zf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
        self.policy_optimizer.step()

        self.fp_optimizer.zero_grad()
        tau[:, :-1].backward(gradient=dWdtau, retain_graph=True)
        entropy_loss.backward()
        fp_grad = nn.utils.clip_grad_norm_(self.fp.parameters(), 1.)
        self.fp_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf, self.target_zf, self.soft_target_tau)
            if self.use_popart:
                ptu.soft_update_from_to(self.zf_normer, self.target_zf_normer, self.soft_target_tau)
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

            self.eval_statistics['ZF Loss'] = zf_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Entropy Loss'] = entropy_loss.item()
            self.eval_statistics['ZF Grad'] = zf_grad
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics['FP Grad'] = fp_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Predictions',
                ptu.get_numpy(z_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Targets',
                ptu.get_numpy(z_target),
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
            self.zf,
            self.target_zf,
        ]
        if self.use_popart:
            networks += [
                self.zf_normer,
                self.target_zf_normer,
            ]
        return networks

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            fp=self.fp,
            target_fp=self.target_fp,
            zf=self.zf,
            target_zf=self.zf,
        )
        if self.use_popart:
            snapshot["zf_normer"] = self.zf_normer
            snapshot["target_zf_normer"] = self.target_zf_normer
        return snapshot
