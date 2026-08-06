import torch.nn as nn

from typing import Callable

from src.sns.mlp import MLP
from src.sns.loss import LipLoss, SNSLoss


class Model(nn.Module):
    def __init__(
        self,
        name: str,
        mlp: MLP,
        loss_fn: Callable | LipLoss | SNSLoss
    ):
        super(Model, self).__init__()

        self.name = name
        self.mlp = mlp
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.mlp(x)

    def get_loss(self, x, y, return_prediction: bool = False):
        y_hat = self.mlp(x)

        bound, curvature = self.mlp.get_bound_and_curvature()

        if isinstance(self.loss_fn, LipLoss):
            loss = self.loss_fn(y, y_hat, bound)
        elif isinstance(self.loss_fn, SNSLoss):
            loss = self.loss_fn(y, y_hat, bound, curvature)
        else:
            loss = self.loss_fn(y, y_hat)

        if return_prediction:
            return loss, y_hat.detach()
        return loss
