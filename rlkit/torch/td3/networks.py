"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import Mlp


def identity(x):
    return x


class TD3Mlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
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
            output_size=1,
            output_activation=output_activation,
            layer_norm=layer_norm,
        )

    def forward(self, state, action):
        h = self.fc1(torch.cat([state, action], dim=1))
        h = torch.cat([h, action], dim=1)
        output = self.fc2(h)
        return output
