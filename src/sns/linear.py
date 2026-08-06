import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWithLipschitz(nn.Module):
    """
    Base class for linear layer modifications with Lipschitz constants.
    """
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.zeros((in_features, out_features))))
        self.b = nn.Parameter(torch.zeros((out_features,)))

    def get_lipschitz_constant(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class SimpleLinear(LinearWithLipschitz):
    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        super().__init__(in_features=in_features, out_features=out_features)

    def get_lipschitz_constant(self):
        abs_row_sum = torch.t(self.W).abs().sum(dim=1)  # (out_features,)
        return torch.max(abs_row_sum)

    def forward(self, x):
        return x @ self.W + self.b


class LipLinear(LinearWithLipschitz):
    """
    Lipschitz Linear from https://arxiv.org/pdf/2202.08345.
    Uses infinity norm based weight normalization for
    satisfying Lipschitz constraint.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_c: float,
        normalization_eps: float = 1e-8
    ):
        super().__init__(in_features=in_features, out_features=out_features)

        assert initial_c > 0
        self.theta = nn.Parameter(self.init_theta(initial_c))
        self.normalization_eps = normalization_eps

    @staticmethod
    def init_theta(c: float):
        return torch.log(torch.exp(torch.tensor([c])) - 1)

    @staticmethod
    def normalize(W, c, eps):
        """
        Normalizes weight matrix such that its infinity norm does not exceed
        provided Lipschitz-constant. Infinity norm of matrix is maximum absolute
        row sum. Normalization includes multiplying each row independently by some scale
        to satisfy the constraint.
        :param W: weight matrix (in_features, out_features).
        :param c: Lipschitz-constant.
        :return: normalized weight matrix.
        """
        abs_row_sum = torch.t(W).abs().sum(dim=1)  # (out_features,)
        scale = torch.minimum(c.to(W.device) / (abs_row_sum + eps), torch.ones_like(abs_row_sum))
        return W * scale[None, :]                  # (in_features, out_features) * (1, out_features)

    def forward(self, x):
        c = self.get_lipschitz_constant()
        return x @ self.normalize(self.W, c, self.normalization_eps) + self.b

    def get_lipschitz_constant(self):
        return F.softplus(self.theta)


class SmoothLinear(LipLinear):
    """
    Linear layer for Smooth Neural Surrogates MLP from https://arxiv.org/pdf/2601.12169.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_c: float,
        normalization_eps: float = 1e-8
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            initial_c=initial_c,
            normalization_eps=normalization_eps
        )

    @staticmethod
    def init_theta(c: float):
        return torch.log(torch.tensor([c]))

    def get_lipschitz_constant(self):
        return torch.exp(self.theta)
