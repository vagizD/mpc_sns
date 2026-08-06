import math
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Tuple, Callable


@dataclass
class CircleBoundaryConfig:
    radius: float = 0.7

@dataclass
class StarBoundaryConfig:
    inner_radius: float = 0.2
    outer_radius: float = 0.7
    num_points: int = 5
    initial_phase: float = math.pi / 2

@dataclass
class TrainingConfig:
    # split shares
    train_share: float = 0.80
    val_share: float = 0.20
    test_share: float = 0.00

    # hyperparameters
    num_epochs: int = 800
    batch_size: int = 8192  # 32768  -- smaller batch size to have more steps per epoch
    num_points_per_shape: int = 500_000  # dataset size

    # CPU/GPU
    num_workers: int = 0
    device: str = "cuda:0"

    # parameters
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    main_loss_fn: Callable = F.mse_loss

    # logging
    path_prefix: str = "/experiment_9[lr=3e-4]"
    save_ckpt_dir: str = "/ckpt"
    save_plots_dir: str = "/plots"

@dataclass
class ModelsConfig:
    hidden_sizes: Tuple[int] = (
        2 + 1,  # raw + latent
        192,
        192,
        192,
        192,
        192,
        1
    )
    act: nn.Module = nn.Softplus

    initial_c: float = 10.0
    normalization_eps: float = 1e-8

    lip_reg_coef: float = 1e-3
    sns1_reg_coef: float = 1e-3
    sns2_reg_coef: float = 1e-3

    sensitivity_budget: float = 8.0

    # computes curvature budget based on sensitivity budget and number of layers
    curvature_budget_fn: Callable = lambda c, L: c * sum([c ** (l / L) for l in range(1, L + 1)])

@dataclass
class InterpolationConfig:
    ts: Tuple[float] = (0, 0.2, 0.4, 0.6, 0.8, 1.0)

    bins: int = 1000  # bins for learned-SDF visualization

@dataclass
class Config:
    CircleCfg = CircleBoundaryConfig
    StarCfg = StarBoundaryConfig

    TrainingCfg = TrainingConfig
    ModelsCfg = ModelsConfig

    InterCfg = InterpolationConfig


cfg = Config()
