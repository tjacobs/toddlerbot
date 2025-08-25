"""PPO configuration settings for ToddlerBot training.

This module defines the PPOConfig dataclass containing hyperparameters and
configuration settings for Proximal Policy Optimization (PPO) training.
"""

from dataclasses import dataclass
from typing import Tuple

import gin


@gin.configurable
@dataclass
class PPOConfig:
    """Data class for storing PPO hyperparameters."""

    wandb_project: str = "ToddlerBot"
    wandb_entity: str = "toddlerbot"
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    use_rnn: bool = False  # specifc to rsl_rl
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
    activation: str = "elu"
    distribution_type: str = "normal"
    noise_std_type: str = "log"
    init_noise_std: float = 0.5
    num_timesteps: int = 500_000_000
    num_evals: int = 100
    episode_length: int = 1000
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantage: bool = True
    normalize_observation: bool = False
    learning_rate: float = 3e-5
    entropy_cost: float = 1e-3
    clipping_epsilon: float = 0.2
    num_envs: int = 1024
    render_nums: int = 20
    batch_size: int = 256
    num_minibatches: int = 4
    seed: int = 0
