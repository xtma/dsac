"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
# from rlkit.torch.core import eval_np
# from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.networks import Mlp


def identity(x):
    return x


def softmax(x):
    return F.softmax(x, dim=-1)


class QuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            num_quantiles=32,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] embedding
        # hidden_sizes[-1] output layer
        self.num_quantiles = num_quantiles
        self.embedding_size = hidden_sizes[-2]
        self.base_fc = Mlp(
            input_size=input_size,
            hidden_sizes=hidden_sizes[:-3],
            output_size=hidden_sizes[-3],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            layer_norm=layer_norm,
        )
        self.tau_fc = Mlp(
            input_size=hidden_sizes[-2],
            hidden_sizes=[],
            output_size=hidden_sizes[-3],
            output_activation=hidden_activation,
        )
        if self.layer_norm:
            self.merge_ln = LayerNorm(hidden_sizes[-3])
        self.merge_fc = Mlp(
            input_size=hidden_sizes[-3],
            hidden_sizes=hidden_sizes[-1:],
            output_size=1,
            hidden_activation=hidden_activation,
            output_activation=identity,
            layer_norm=layer_norm,
        )
        self.last_fc = self.merge_fc.last_fc

    def forward(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * ptu.from_numpy(np.arange(1, 1 + self.embedding_size)) * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        if self.layer_norm:
            h = self.merge_ln(h)
        output = self.merge_fc(h).squeeze(-1)
        return output


class CatQuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            obs_dim,
            action_dim,
            num_quantiles=32,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] embedding
        # hidden_sizes[-1] output layer
        assert len(hidden_sizes) == 4
        self.num_quantiles = num_quantiles
        self.embedding_size = hidden_sizes[2]
        self.fc1 = Mlp(
            input_size=obs_dim + action_dim,
            hidden_sizes=[],
            output_size=hidden_sizes[0],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            layer_norm=layer_norm,
        )
        self.fc2 = Mlp(
            input_size=hidden_sizes[0] + action_dim,
            hidden_sizes=[],
            output_size=hidden_sizes[1],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            layer_norm=layer_norm,
        )
        self.tau_fc = Mlp(
            input_size=hidden_sizes[2],
            hidden_sizes=[],
            output_size=hidden_sizes[1],
            output_activation=hidden_activation,
        )
        if self.layer_norm:
            self.merge_ln = LayerNorm(hidden_sizes[1])
        self.merge_fc = Mlp(
            input_size=hidden_sizes[1],
            hidden_sizes=hidden_sizes[-1:],
            output_size=1,
            hidden_activation=hidden_activation,
            output_activation=identity,
            layer_norm=layer_norm,
        )
        self.last_fc = self.merge_fc.last_fc

    def forward(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.fc1(h)  # (N, C)
        h = torch.cat([h, action], dim=1)
        h = self.fc2(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * ptu.from_numpy(np.arange(1, 1 + self.embedding_size)) * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        if self.layer_norm:
            h = self.merge_ln(h)
        output = self.merge_fc(h).squeeze(-1)
        return output


class CatMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()
        self.fc1 = Mlp(
            input_size=obs_dim + action_dim,
            hidden_sizes=[],
            output_size=hidden_sizes[0],
            output_activation=hidden_activation,
            layer_norm=layer_norm,
        )
        self.fc2 = Mlp(
            input_size=action_dim + hidden_sizes[0],
            hidden_sizes=hidden_sizes[1:],
            output_size=output_size,
            output_activation=output_activation,
            layer_norm=layer_norm,
        )

    def forward(self, state, action):
        h = self.fc1(torch.cat([state, action], dim=1))
        h = torch.cat([h, action], dim=1)
        output = self.fc2(h)
        return output


class PopArt(nn.Module):

    def __init__(self, output_layer, beta: float = 0.0001, zero_debias: bool = False, start_pop: int = 0):
        # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
        super(PopArt, self).__init__()
        self.start_pop = start_pop
        self.zero_debias = zero_debias
        self.beta = beta
        self.output_layers = output_layer if isinstance(output_layer, (tuple, list, nn.ModuleList)) else (output_layer,)
        shape = self.output_layers[0].bias.shape
        device = self.output_layers[0].bias.device
        assert all(shape == x.bias.shape for x in self.output_layers)
        self.mean = nn.Parameter(torch.zeros(shape, device=device), requires_grad=False)
        self.mean_square = nn.Parameter(torch.ones(shape, device=device), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape, device=device), requires_grad=False)
        self.updates = 0

    def forward(self, *input):
        pass

    @torch.no_grad()
    def update(self, targets):
        targets_shape = targets.shape
        targets = targets.view(-1, 1)
        beta = max(1. / (self.updates + 1.), self.beta) if self.zero_debias else self.beta
        # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data
        new_mean = (1. - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1. - beta) * self.mean_square + beta * (targets * targets).mean(0)
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)
        assert self.std.shape == (1,), 'this has only been tested in 1D'
        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= self.std / new_std
                layer.bias *= self.std
                layer.bias += self.mean - new_mean
                layer.bias /= new_std
        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.norm(targets).view(*targets_shape)

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, value):
        return value * self.std + self.mean
