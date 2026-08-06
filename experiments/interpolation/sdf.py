import torch
from typing import Tuple


class SDFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        points: torch.Tensor,
        interpolation_coef: torch.Tensor,
        signed_distance: torch.Tensor
    ):
        assert points.ndim == 2 and signed_distance.ndim == 2 and interpolation_coef.ndim == 2
        assert points.shape[0] == signed_distance.shape[0] and points.shape[0] == interpolation_coef.shape[0]
        assert points.shape[1] == 2
        assert interpolation_coef.shape[1] == 1
        assert signed_distance.shape[1] == 1

        self.points = points
        self.interpolation_coef = interpolation_coef
        self.signed_distance = signed_distance

    def __len__(self):
        return self.points.size(0)

    def __getitem__(self, idx):
        points = self.points[idx]
        inter_coef = self.interpolation_coef[idx]

        # sin = torch.sin(points)
        # cos = torch.cos(points)
        # features = torch.cat([points, sin, cos, inter_coef], dim=-1)
        features = torch.cat([points, inter_coef], dim=-1)
        return features, self.signed_distance[idx]


def generate_points(
    x_lim: Tuple[float, float] = (-1, 1),
    y_lim: Tuple[float, float] = (-1, 1),
    bins: int = None,
    num_points: int = None
) -> torch.Tensor:
    """
    Generates points from rectangle.
    :x_lim: x-axis values limit in form (min, max).
    :y_lim: y-axis values limit in form (min, max).
    :bins: (for lattice mode): number of bins to split each axis into.
    :num_points: (for random mode): number of points to sample.
    :return: generated points of shape (N, 2). For lattice mode N = bins * bins,
        for random mode N = num_points.
    """
    is_lattice = bins is not None
    is_random = num_points is not None

    assert is_lattice or is_random
    assert not is_lattice or not is_random

    assert x_lim[0] < x_lim[1]
    assert y_lim[0] < y_lim[1]

    if is_lattice:
        assert bins > 1

        xs = torch.linspace(x_lim[0], x_lim[1], steps=bins)
        ys = torch.linspace(y_lim[0], y_lim[1], steps=bins)

        grid_x, grid_y = torch.meshgrid([xs, ys], indexing='ij')
        x, y = grid_x.flatten(), grid_y.flatten()
        points = torch.stack([x, y], dim=-1)

    elif is_random:
        assert num_points > 0

        points_raw = torch.rand((num_points, 2))  # x, y
        p_min = torch.tensor([x_lim[0], y_lim[0]])
        p_max = torch.tensor([x_lim[1], y_lim[1]])

        points = points_raw * (p_max - p_min) + p_min  # [0, 1) -> [p_min, p_max]
    else:
        raise RuntimeError("Incorrect sampling arguments.")
    return points
