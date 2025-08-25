"""Cartwheel motion reference implementation for toddlerbot.

Provides cartwheel motion references using precomputed motion data.
"""

import os
from typing import Dict, Tuple

import jax
import joblib

from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class CartwheelReference(MotionReference):
    """Motion reference for cartwheel movements using precomputed motion data."""

    def __init__(self, robot, dt: float, fixed_base: bool = False):
        """Initialize cartwheel motion reference.

        Args:
            robot: Robot instance for motion generation.
            dt: Time step for motion reference.
            fixed_base: Whether to use fixed base mode.
        """
        super().__init__("cartwheel", "human", robot, dt, fixed_base)

        # Load cartwheeling motion data
        robot_suffix = "_2xc" if "2xc" in robot.name else "_2xm"
        motion_file_path = os.path.join("motion", f"cartwheel{robot_suffix}.lz4")
        self.motion_ref = joblib.load(motion_file_path)

        for key in ["keyframes", "timed_sequence"]:
            if key in self.motion_ref:
                del self.motion_ref[key]

        # Extract data from the cartwheel.lz4 format
        self.time_arr = np.array(self.motion_ref["time"])
        self.n_frames = len(self.time_arr)

        # Check if motion data is in robot-relative frame (default: False for backward compatibility)
        self.is_robot_relative_frame = self.motion_ref.get(
            "is_robot_relative_frame", False
        )
        frame_type = "relative" if self.is_robot_relative_frame else "global"

        print("\n=== MOTION DATA ===")
        print(f"  - path     : {motion_file_path}")
        print(f"  - qpos     : {self.motion_ref['qpos'].shape}")
        print(f"  - body_pos : {self.motion_ref['body_pos'].shape}")
        print(f"  - site_pos : {self.motion_ref['site_pos'].shape}")
        print(f"  - frame    : {frame_type}")

        if self.use_jax:
            # Keep large arrays on CPU, only convert small arrays to JAX
            self.time_arr = jax.device_put(self.time_arr)
            self.motion_ref = jax.device_put(self.motion_ref)

    def get_phase_signal(self, time_curr: float, init_idx: int = 0) -> ArrayType:
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % self.n_frames

        # Calculate phase based on total_idx
        phase = (total_idx / self.n_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Get the desired linear and angular velocities."""
        lin_vel = np.zeros(3, dtype=np.float32)  # No linear velocity
        ang_vel = np.zeros(3, dtype=np.float32)  # No rotation
        return lin_vel, ang_vel

    def get_state_ref(
        self,
        time_curr: float,
        command: ArrayType,
        last_state: Dict[str, ArrayType],
        init_idx: int = 0,
    ) -> Dict[str, ArrayType]:
        """Get the reference state for the current time. Supports RIS if fed init_idx

        Args:
            time_curr (float): The current time.
            command (ArrayType): Command inputs for the robot's movement.
            last_state (Dict[str, ArrayType]): The last state of the robot.
            init_idx (int, optional): Starting initial state index for RIS. Defaults to 0.

        Returns:
            Dict[str, ArrayType]: A dictionary containing the path state, motor positions, joint positions, body poses, and other reference data.
        """
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        # Cartwheel motion is not periodic
        curr_idx = np.min(np.array([init_idx + time_idx, self.n_frames - 1]))
        # Get reference qpos from keyframes
        qpos = self.motion_ref["qpos"][curr_idx]
        if self.fixed_base:
            qpos = qpos[7:]  # Skip first 7 elements for fixed base

        joint_pos = qpos[self.q_start_idx + self.mj_joint_indices]
        # Get motor positions from action data
        motor_pos = qpos[self.q_start_idx + self.mj_motor_indices]

        # OPTIMIZATION: Use direct indexing for body poses (like walk_zmp_ref)
        body_pos = self.motion_ref["body_pos"][curr_idx]
        body_quat = self.motion_ref["body_quat"][curr_idx]
        body_lin_vel = self.motion_ref["body_lin_vel"][curr_idx]
        body_ang_vel = self.motion_ref["body_ang_vel"][curr_idx]

        # Get reference site poses
        site_pos = self.motion_ref["site_pos"][curr_idx]
        site_quat = self.motion_ref["site_quat"][curr_idx]
        stance_mask = np.ones(2, dtype=np.float32)

        # Get path state using the new interface
        path_state = self.integrate_path_state(command, last_state)

        return {
            **path_state,
            "motor_pos": motor_pos,
            "joint_pos": joint_pos,
            "qpos": qpos,
            "body_pos": body_pos,
            "body_quat": body_quat,
            "body_lin_vel": body_lin_vel,
            "body_ang_vel": body_ang_vel,
            "site_pos": site_pos,
            "site_quat": site_quat,
            "stance_mask": stance_mask,
        }
