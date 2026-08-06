import torch.nn as nn
from typing import Tuple, Type

from src.sns.linear import LinearWithLipschitz


class MLP(nn.Module):
    def __init__(
        self,
        linear_cls: Type[LinearWithLipschitz],
        hidden_sizes: Tuple[int],
        act: nn.Module,
        **kwargs
    ):
        super(MLP, self).__init__()

        self.linear_layers = nn.ModuleList()
        self.act = act()
        for l in range(len(hidden_sizes) - 1):
            self.linear_layers.append(linear_cls(
                in_features=hidden_sizes[l],
                out_features=hidden_sizes[l + 1],
                **kwargs
            ))

    def forward(self, x):
        for layer in self.linear_layers[:-1]:
            x = self.act(layer(x))
        return self.linear_layers[-1](x)

    def get_bound_and_curvature(self):
        bound = self.linear_layers[0].get_lipschitz_constant()
        curvature = bound.clone()
        for layer in self.linear_layers[1:]:
            bound = bound * layer.get_lipschitz_constant()
            curvature = curvature + bound
        return bound, curvature
