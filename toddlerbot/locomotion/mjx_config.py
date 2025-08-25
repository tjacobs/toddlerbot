"""Configuration classes for MJX environments.

This module defines configuration dataclasses for MJX-based locomotion environments,
including simulation parameters, terrain settings, observation configurations,
rewards, actions, domain randomization, and noise settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gin


@gin.configurable
@dataclass
class MJXConfig:
    """Configuration class for the MJX environment."""

    @gin.configurable
    @dataclass
    class SimConfig:
        timestep: float = 0.005
        solver: int = 2  # Newton
        # A list of explicit geom pairs for self collision checks.
        # If None, the default self contact pairs defined in the robot MJCF will be used.
        self_contact_pairs: Optional[List[List[str]]] = field(default=None)
        # Example:
        # self_contact_pairs: List[List[str]] = field(
        #     default_factory=lambda: [
        #         ["left_ankle_roll_link_collision", "right_ankle_roll_link_collision"],
        #         ["left_hand_collision", "right_hand_collision"],
        #     ]
        # )
        # impratio: int = 10
        # max_contact_points: int = 20
        # max_geom_pairs: int = 5

    @gin.configurable
    @dataclass
    class TerrainConfig:
        tile_width: float = 4.0
        tile_length: float = 4.0
        resolution_per_meter: int = 16
        random_spawn: bool = False
        manual_map: List[List[str]] = field(
            default_factory=lambda: [
                ["flat"]
                # ["boxes", "stairs", "bumps"],
                # ["flat", "slope", "rough"],
            ]
        )
        # random_spawn: bool = True
        # manual_map: List[List[str]] = field(
        #     default_factory=lambda: [
        #         ["flat", "rough", "bumps"],
        #         ["rough", "bumps", "flat"],
        #         ["bumps", "flat", "rough"],
        #     ]
        # )
        robot_collision_geom_names: Optional[List[str]] = field(
            default_factory=lambda: [
                "left_ankle_roll_link_collision",
                "right_ankle_roll_link_collision",
                "left_hand_collision",
                "right_hand_collision",
            ]
        )
        # Optional list of robot geom names to be used for contact with terrain.
        # If not provided, default contact pairs from the robot XML file will be used.

    @gin.configurable
    @dataclass
    class ObsConfig:
        frame_stack: int = 15
        c_frame_stack: int = 15
        num_single_obs: int = 84
        num_single_privileged_obs: int = 151

    @gin.configurable
    @dataclass
    class ObsScales:
        lin_vel: float = 2.0
        ang_vel: float = 1.0
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        quat: float = 1.0  # Now used for quaternion scaling
        actuator_force: float = 0.1
        # height_measurements: float = 5.0

    @gin.configurable
    @dataclass
    class ActionConfig:
        action_parts: List[str] = field(default_factory=lambda: ["leg"])
        action_scale: float = 0.25
        contact_force_threshold: float = 1.0
        n_steps_delay: int = 1
        n_frames: int = 4
        cycle_time: float = 0.72

    @gin.configurable
    @dataclass
    class RewardsConfig:
        healthy_z_range: List[float] = field(default_factory=lambda: [0.2, 1.0])
        pos_tracking_sigma: float = 200.0
        rot_tracking_sigma: float = 20.0
        lin_vel_tracking_sigma: float = 200.0
        ang_vel_tracking_sigma: float = 0.5
        min_feet_y_dist: float = 0.07
        max_feet_y_dist: float = 0.13
        torso_roll_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        torso_pitch_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])
        add_regularization: bool = True
        use_exp_reward: bool = True
        # Body part weights for tracking rewards (motor position, pose, etc.)
        leg_weight: float = 1.0
        arm_weight: float = 1.0
        neck_weight: float = 0.2
        waist_weight: float = 0.2

    @gin.configurable
    @dataclass
    class RewardScales:
        torso_pos_xy: float | Dict[str, float] = 0.0
        torso_pos_z: float | Dict[str, float] = 0.0
        torso_quat: float | Dict[str, float] = 0.0
        torso_roll: float | Dict[str, float] = 0.0
        torso_pitch: float | Dict[str, float] = 0.0
        lin_vel_xy: float | Dict[str, float] = 0.0
        lin_vel_z: float | Dict[str, float] = 0.0
        ang_vel_xy: float | Dict[str, float] = 0.0
        ang_vel_z: float | Dict[str, float] = 0.0
        motor_pos: float | Dict[str, float] = 0.0  # Combined motor position reward
        motor_torque: float | Dict[str, float] = 0.0
        energy: float | Dict[str, float] = 0.0
        action_rate: float | Dict[str, float] = 0.0
        feet_contact: float | Dict[str, float] = 0.0
        contact_number: float | Dict[str, float] = 0.0
        collision: float | Dict[str, float] = 0.0
        survival: float | Dict[str, float] = 0.0
        feet_air_time: float | Dict[str, float] = 0.0
        feet_distance: float | Dict[str, float] = 0.0
        feet_slip: float | Dict[str, float] = 0.0
        feet_clearance: float | Dict[str, float] = 0.0
        stand_still: float | Dict[str, float] = 0.0
        align_ground: float | Dict[str, float] = 0.0
        body_quat: float | Dict[str, float] = 0.0
        site_pos: float | Dict[str, float] = 0.0
        body_lin_vel: float | Dict[str, float] = 0.0
        body_ang_vel: float | Dict[str, float] = 0.0

        def reset(self):
            """Reset all reward scales to zero."""
            for key in vars(self):
                setattr(self, key, 0.0)

    @gin.configurable
    @dataclass
    class CommandsConfig:
        resample_time: float = 3.0
        zero_chance: float = 0.2
        turn_chance: float = 0.2
        command_obs_indices: List[int] = field(default_factory=lambda: [])
        command_range: List[List[float]] = field(default_factory=lambda: [[]])
        deadzone: List[float] = field(default_factory=lambda: [])

    @gin.configurable
    @dataclass
    class DomainRandConfig:
        add_domain_rand: bool = True
        # [0] is the default initial state, [] enables all possible frames as initial states
        rand_init_state_indices: List[int] = field(default_factory=lambda: [0])
        add_push: bool = False
        add_head_pose: bool = False
        backlash_activation: float = 0.1
        backlash_range: List[float] = field(default_factory=lambda: [0.02, 0.1])
        torso_roll_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        torso_pitch_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        arm_joint_pos_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        friction_range: List[float] = field(default_factory=lambda: [0.4, 1.0])
        damping_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        armature_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        frictionloss_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        body_mass_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])
        hand_mass_range: List[float] = field(default_factory=lambda: [0.0, 0.1])
        other_mass_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
        kp_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        kd_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        tau_max_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        q_dot_tau_max_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        q_dot_max_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        kd_min_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        tau_brake_max_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        tau_q_dot_max_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
        passive_active_ratio_range: List[float] = field(
            default_factory=lambda: [0.5, 1.5]
        )
        push_interval_s: float = 2.0  # seconds
        push_duration_s: float = 0.2  # seconds
        push_torso_range: List[float] = field(default_factory=lambda: [1.0, 3.0])
        push_other_range: List[float] = field(default_factory=lambda: [1.0, 3.0])

    @gin.configurable
    @dataclass
    class NoiseConfig:
        level: float = 0.05
        dof_pos: float = 1.0
        dof_vel: float = 2.0
        # gyro (rad/s)
        gyro_fc: float = 0.35  # low cutoff -> slow wander
        gyro_std: float = 0.25  # steady-state std (rad/s)
        gyro_bias_walk_std: float = 2e-4  # per-step RW std
        gyro_white_std: float = 0.0
        # quaternion small-angle rotvec (rad)
        quat_fc: float = 0.25
        quat_std: float = 0.10
        quat_bias_walk_std: float = 1e-4
        quat_white_std: float = 0.0
        gyro_amp_min: float = 0.8
        gyro_amp_max: float = 2.0
        quat_amp_min: float = 0.8
        quat_amp_max: float = 1.2

    def __init__(self):
        """Initialize all configuration sections with their default values."""
        self.sim = self.SimConfig()
        self.terrain = self.TerrainConfig()
        self.obs = self.ObsConfig()
        self.obs_scales = self.ObsScales()
        self.action = self.ActionConfig()
        self.rewards = self.RewardsConfig()
        self.reward_scales = self.RewardScales()
        self.commands = self.CommandsConfig()
        self.domain_rand = self.DomainRandConfig()
        self.noise = self.NoiseConfig()
