"""RSL-RL wrapper for MJX environments.

This module provides the RSLRLWrapper class that adapts MJX environments
for use with the RSL-RL reinforcement learning framework, handling
JAX-PyTorch tensor conversions and environment interface compatibility.
"""

import jax
import jax.numpy as jnp
import torch
from brax.io.torch import jax_to_torch, torch_to_jax
from rsl_rl.env.vec_env import VecEnv

from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.ppo_config import PPOConfig

# from toddlerbot.utils.misc_utils import profile


class RSLRLWrapper(VecEnv):
    """Wrapper to adapt MJX environments for RSL-RL training framework."""

    def __init__(
        self,
        env: MJXEnv,
        device: torch.device,
        train_cfg: PPOConfig,
        eval: bool = False,
    ):
        """Initialize the wrapper with environment setup and device configuration."""
        self.env = env
        self.device = device
        self.num_actions = env.action_size
        self.cfg = env.cfg

        self.num_obs = env.obs_size

        if eval:
            self.num_envs = 1
            self.max_episode_length = train_cfg.episode_length
            self.key_envs = jax.random.PRNGKey(train_cfg.seed)
        else:
            self.num_envs = train_cfg.num_envs
            self.max_episode_length = train_cfg.episode_length
            key = jax.random.PRNGKey(train_cfg.seed)
            self.key_envs = jax.random.split(key, self.num_envs)

        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.global_step_buf = jnp.zeros(self.num_envs, dtype=jnp.int32)

        self.reset_fn = jax.jit(env.reset)
        self.step_fn = jax.jit(env.step)

        self.last_state = self.reset()

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Get current observations and convert from JAX to PyTorch format."""
        obs = self.last_state.obs["state"]
        privileged_obs = self.last_state.obs["privileged_state"]
        obs_torch = jax_to_torch(obs, device=self.device)
        privileged_obs_torch = jax_to_torch(privileged_obs, device=self.device)
        if len(obs_torch.shape) == 1:
            obs_torch = obs_torch.unsqueeze(0)
            privileged_obs_torch = privileged_obs_torch.unsqueeze(0)

        return obs_torch, {"observations": {"critic": privileged_obs_torch}}

    def reset(self):
        """Reset all environments and return initial states."""
        return self.reset_fn(self.key_envs)

    # @profile()
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute actions in environment and return step results."""
        actions_jax = torch_to_jax(actions)
        state_curr = self.step_fn(self.last_state, actions_jax)

        state_jax = {
            "obs": state_curr.obs["state"],
            "priv_obs": state_curr.obs["privileged_state"],
            "reward": state_curr.reward,
            "done": state_curr.done,
            "metrics": state_curr.info["episode_metrics"],
            "episode_done": state_curr.info["episode_done"],
            "truncation": state_curr.info["truncation"],
        }

        self.global_step_buf = jnp.where(
            state_curr.done,
            self.global_step_buf + state_curr.info["episode_metrics"]["length"],
            self.global_step_buf,
        )
        state_curr.info["global_step"] = jnp.repeat(
            jnp.sum(self.global_step_buf), self.num_envs
        )

        state_torch = jax_to_torch(state_jax, device=self.device)

        obs = state_torch["obs"]
        privileged_obs = state_torch["priv_obs"]
        rewards = state_torch["reward"]
        dones = state_torch["done"]
        infos = {
            "observations": {"critic": privileged_obs},
            "episode": state_torch["metrics"],
            "dones": state_torch["episode_done"],
            "time_outs": state_torch["truncation"],
        }

        self.last_state = state_curr

        return obs, rewards, dones, infos
