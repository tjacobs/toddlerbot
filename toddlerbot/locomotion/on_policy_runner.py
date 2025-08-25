"""On-policy reinforcement learning runner for ToddlerBot training.

This module provides the OnPolicyRunner class for training and evaluating policies
using on-policy algorithms like PPO. It manages the training loop, checkpointing,
logging, and policy evaluation with support for distributed training.
"""

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import os
import time

import numpy as np
import torch
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)

# from toddlerbot.utils.misc_utils import profile


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        run_name: str | None = None,
        progress_fn: callable | None = None,
        render_fn: callable | None = None,
        restore_params=None,
        device="cpu",
    ):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(
                f"Training type not found for algorithm {self.alg_cfg['class_name']}."
            )

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = (
                    "critic"  # actor-critic reinforcement learnig, e.g., PPO
                )
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[
                1
            ]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        self.class_name = self.policy_cfg.pop("class_name")
        policy_class = eval(self.class_name)
        policy: (
            ActorCritic
            | ActorCriticRecurrent
            | StudentTeacher
            | StudentTeacherRecurrent
        ) = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError(
                    "Observations for the key 'rnd_state' not found in infos['observations']."
                )
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.render_interval = self.cfg["render_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(
                self.device
            )  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(
                self.device
            )  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.run_name = run_name
        self.log_dir = (
            os.path.join("results", run_name) if run_name is not None else None
        )
        if self.log_dir:
            os.makedirs(os.path.join(self.log_dir, "ckpt"), exist_ok=True)

        self.current_learning_iteration = 0
        self.best_ckpt = 0
        self.last_ckpt = 0
        self.best_mean_reward = -float("inf")
        self.last_mean_reward = 0.0

        self.steps_between_logging = self.env.num_envs * self.num_steps_per_env
        self.metrics_buffer = collections.defaultdict(
            lambda: collections.deque(maxlen=100)
        )
        self.num_steps = 0
        self.last_log_steps = 0
        self.log_count = 0
        self.progress_fn = progress_fn
        self.render_fn = render_fn

        if restore_params:
            self.load(restore_params)

    # @profile()
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError(
                "Teacher model parameters not loaded. Please load a teacher model to distill."
            )

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        self.time_start = time.monotonic()
        info_loss = {}
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(
                                self.device
                            )
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # book keeping
                    self.update_episode_metrics(
                        infos["episode"], infos["dones"], info_loss
                    )

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

                if self.render_fn and it and it % self.render_interval == 0:
                    # Use the MJX env to render
                    self.render_fn(self.get_inference_policy(device=self.device), it)
                    if "Recurrent" in self.class_name:
                        self.train_mode()

            # update policy
            loss_dict = self.alg.update()
            info_loss["loss/policy"] = loss_dict["surrogate"]
            info_loss["loss/value_func"] = (
                loss_dict["value_function"] * self.alg.value_loss_coef
            )
            info_loss["loss/entropy"] = loss_dict["entropy"] * -self.alg.entropy_coef
            info_loss["loss/total"] = (
                info_loss["loss/policy"]
                + info_loss["loss/value_func"]
                + info_loss["loss/entropy"]
            )
            info_loss["loss/learning_rate"] = self.alg.learning_rate
            info_loss["loss/mean_noise_std"] = self.alg.policy.action_std.mean()
            info_loss["loss/return"] = self.alg.storage.returns.mean()
            info_loss["loss/advantage"] = self.alg.storage.advantages.mean()
            info_loss["loss/value"] = self.alg.storage.values.mean()

            total_norm = 0.0
            for p in self.alg.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            info_loss["loss/grad_norm"] = total_norm

            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "ckpt", f"model_{it}.pt"))
                    self.last_ckpt = it

    def update_episode_metrics(self, metrics, dones, info_loss):
        """Update episode metrics and log training progress."""
        self.num_steps += dones.numel()
        if torch.sum(dones) > 0:
            for name, metric in metrics.items():
                done_metrics = metric[dones.bool()].flatten().tolist()
                self.metrics_buffer[name].extend(done_metrics)

        if self.num_steps - self.last_log_steps >= self.steps_between_logging:
            self.log_count += 1
            mean_metrics = {}
            for metric_name in self.metrics_buffer:
                mean_metrics[metric_name] = np.mean(self.metrics_buffer[metric_name])

            if "sum_reward" in mean_metrics:
                self.last_mean_reward = float(mean_metrics["sum_reward"])

            if self.progress_fn is not None:
                log_data = {
                    f"episode/{name}": value for name, value in mean_metrics.items()
                }
                log_data.update(info_loss)
                log_data["sps"] = self.num_steps / (time.monotonic() - self.time_start)
                self.progress_fn(int(self.num_steps), log_data)

            self.last_log_steps = self.num_steps

    def save(self, path: str, infos=None):
        """Save model checkpoint including policy, optimizer, and training state."""
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )

        # save model
        torch.save(saved_dict, path)

        if self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            self.best_ckpt = self.current_learning_iteration

    def load(self, loaded_dict: dict, load_optimizer: bool = True):
        """Load model checkpoint and optionally resume training state."""
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(
            loaded_dict["model_state_dict"]
        )
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["privileged_obs_norm_state_dict"]
                )
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["obs_norm_state_dict"]
                )
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(
                    loaded_dict["rnd_optimizer_state_dict"]
                )
        # -- load current learning iteration
        if resumed_training and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]

        # return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """Get inference policy for evaluation with optional normalization."""
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        """Set all models to training mode."""
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """Set all models to evaluation mode."""
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
