import torch
import torch.nn.functional as F
from typing import Callable


class LipLoss:
    def __init__(
        self,
        regularization_coef: float,
        main_loss: Callable = F.mse_loss
    ):
        assert regularization_coef > 0.0

        self.regularization_coef = regularization_coef
        self.main_loss = main_loss

    def __call__(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        bound: torch.Tensor,
    ):
        main_loss = self.main_loss(input, target)
        reg_loss = bound
        loss = main_loss + self.regularization_coef * reg_loss
        return loss


class SNSLoss:
    def __init__(
        self,
        k: int,
        budget: float,
        regularization_coef: float,
        main_loss: Callable = F.mse_loss
    ):
        assert k in (1, 2)
        assert budget > 0.0
        assert regularization_coef > 0.0

        self.k = k
        self.budget = budget
        self.regularization_coef = regularization_coef
        self.main_loss = main_loss

    def __call__(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        bound: torch.Tensor,
        curvature: torch.Tensor
    ):
        main_loss = self.main_loss(input, target)

        lipschitz_penalty = bound * torch.pow(curvature, self.k - 1)
        reg_loss = torch.clamp(lipschitz_penalty / self.budget, min=torch.ones_like(lipschitz_penalty))

        loss = main_loss + self.regularization_coef * reg_loss
        return loss
