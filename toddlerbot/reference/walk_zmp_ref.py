"""ZMP-based walking motion reference implementation for toddlerbot.

Provides walking motion references using Zero Moment Point (ZMP) control with lookup tables.
"""

import os
from typing import Dict, Tuple

import jax
import joblib

from toddlerbot.algorithms.zmp_walk import ZMPWalk
from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, R, inplace_update
from toddlerbot.utils.array_utils import array_lib as np

# from toddlerbot.utils.misc_utils import profile


class WalkZMPReference(MotionReference):
    """Class for generating a ZMP-based walking reference for the toddlerbot robot."""

    def __init__(
        self, robot: Robot, dt: float, cycle_time: float, fixed_base: bool = False
    ):
        """Initializes the walk ZMP (Zero Moment Point) controller.

        Args:
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for the controller.
            cycle_time (float): The duration of one walking cycle.
        """
        super().__init__("walk_zmp", "periodic", robot, dt, fixed_base)

        self.cycle_time = cycle_time

        robot_suffix = "_2xc" if "2xc" in robot.name else "_2xm"
        lookup_table_path = os.path.join("motion", f"walk_zmp{robot_suffix}.lz4")
        if os.path.exists(lookup_table_path):
            lookup_keys, motion_ref_list = joblib.load(lookup_table_path)
        else:
            zmp_walk = ZMPWalk(self.robot, self.cycle_time, 2.0)
            lookup_keys, motion_ref_list = zmp_walk.build_lookup_table(robot_suffix)
            joblib.dump(
                (lookup_keys, motion_ref_list), lookup_table_path, compress="lz4"
            )

        self.episode_len = motion_ref_list[0]["qpos"].shape[0]
        self.lookup_keys = np.array(lookup_keys, dtype=np.float32)
        self.motion_lookup = {}
        for k in motion_ref_list[0]:
            self.motion_lookup[k] = np.stack([m[k] for m in motion_ref_list])

        if self.robot.has_gripper:
            shape = self.motion_lookup["qpos"].shape[:2]
            self.motion_lookup["qpos"] = inplace_update(
                np.tile(self.default_qpos[self.q_start_idx :], (*shape, 1)),
                (slice(None), slice(None), ~self.mj_gripper_mask),
                self.motion_lookup["qpos"],
            )

        if self.use_jax:
            self.lookup_keys = jax.device_put(self.lookup_keys)
            # Convert each array in the dictionary separately to avoid vmap issues
            self.motion_lookup = {
                k: jax.device_put(v) for k, v in self.motion_lookup.items()
            }

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        """Calculate the phase signal as a sine and cosine pair for a given time.

        Args:
            time_curr (float | ArrayType): The current time or an array of time values.

        Returns:
            ArrayType: A numpy array containing the sine and cosine of the phase, with dtype float32.
        """
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Calculates linear and angular velocities based on the given command array.

        The function interprets the command array to extract linear and angular velocity
        components. The linear velocity is derived from specific indices of the command
        array, while the angular velocity is determined from another index.

        Args:
            command (ArrayType): An array containing control commands, where indices 5 and 6
                are used for linear velocity components and index 7 for angular velocity.

        Returns:
            Tuple[ArrayType, ArrayType]: A tuple containing the linear velocity as the first
            element and the angular velocity as the second element, both as numpy arrays.
        """
        # The first 5 commands are neck yaw, neck pitch, arm, waist roll, waist yaw
        lin_vel = np.array([command[5], command[6], 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, command[7]], dtype=np.float32)
        return lin_vel, ang_vel

    # @profile()
    def get_state_ref(
        self,
        time_curr: float | ArrayType,
        command: ArrayType,
        last_state: Dict[str, ArrayType],
        init_idx: int = 0,
        torso_yaw: float = 0.0,
    ) -> Dict[str, ArrayType]:
        """Generate a reference state for a robotic system based on the current state, time, and command inputs.

        This function calculates the desired joint and motor positions for a robot by integrating the current state with the given command inputs. It interpolates neck, waist, and arm positions, and determines leg joint positions based on whether the robot is in a static pose or dynamic motion. The function returns a concatenated array representing the path state, motor positions, joint positions, and stance mask.

        Args:
            time_curr (float | ArrayType): The current time or time array.
            command (ArrayType): Command inputs for the robot's movement.
            last_state (ArrayType): The current state of the robot.

        Returns:
            Dict[str, ArrayType]: A dictionary containing the path state, motor positions, joint positions, and stance mask.
        """
        path_state = self.integrate_path_state(command, last_state)
        is_static = np.logical_or(np.linalg.norm(command[5:]) < 1e-6, time_curr < 1e-6)

        heading_rot = R.from_euler("z", torso_yaw)
        heading_error = heading_rot.inv() * path_state["path_rot"]
        lin_vel_correct = heading_error.apply(
            np.concatenate([command[5:7], np.zeros(1)])
        )
        command_correct = np.concatenate([lin_vel_correct[:2], command[7:]])
        nearest_command_idx = np.argmin(
            np.linalg.norm(self.lookup_keys - command_correct, axis=1)
        )
        step_idx = (np.round(time_curr / self.dt) % self.episode_len).astype(int)

        qpos = np.where(
            is_static,
            self.default_qpos[self.q_start_idx :],
            self.motion_lookup["qpos"][nearest_command_idx][step_idx],
        )
        joint_pos = qpos[self.mj_joint_indices]
        motor_pos = qpos[self.mj_motor_indices]

        neck_yaw_pos = np.interp(
            command[0],
            np.array([-1, 0, 1]),
            np.array(
                [
                    self.robot.neck_joint_limits[0, 0],
                    0.0,
                    self.robot.neck_joint_limits[1, 0],
                ]
            ),
        )
        neck_pitch_pos = np.interp(
            command[1],
            np.array([-1, 0, 1]),
            np.array(
                [
                    self.robot.neck_joint_limits[0, 1],
                    0.0,
                    self.robot.neck_joint_limits[1, 1],
                ]
            ),
        )
        neck_joint_pos = np.array([neck_yaw_pos, neck_pitch_pos])
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.neck_motor_indices, self.robot.neck_ik(neck_joint_pos)
        )

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = np.where(
            command[2] > 0,
            self.arm_joint_pos_ref[ref_idx],
            self.default_joint_pos[self.arm_joint_indices],
        )
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.arm_motor_indices, self.robot.arm_ik(arm_joint_pos)
        )

        waist_roll_pos = np.interp(
            command[3],
            np.array([-1, 0, 1]),
            np.array(
                [
                    self.robot.waist_joint_limits[0, 0],
                    0.0,
                    self.robot.waist_joint_limits[1, 0],
                ]
            ),
        )
        waist_yaw_pos = np.interp(
            command[4],
            np.array([-1, 0, 1]),
            np.array(
                [
                    self.robot.waist_joint_limits[0, 1],
                    0.0,
                    self.robot.waist_joint_limits[1, 1],
                ]
            ),
        )
        waist_joint_pos = np.array([waist_roll_pos, waist_yaw_pos])
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.waist_motor_indices, self.robot.waist_ik(waist_joint_pos)
        )

        body_pos = self.motion_lookup["body_pos"][nearest_command_idx][step_idx]
        body_quat = self.motion_lookup["body_quat"][nearest_command_idx][step_idx]
        body_lin_vel = self.motion_lookup["body_lin_vel"][nearest_command_idx][step_idx]
        body_ang_vel = self.motion_lookup["body_ang_vel"][nearest_command_idx][step_idx]
        site_pos = self.motion_lookup["site_pos"][nearest_command_idx][step_idx]
        site_quat = self.motion_lookup["site_quat"][nearest_command_idx][step_idx]
        stance_mask = self.motion_lookup["contact"][nearest_command_idx][step_idx]
        stance_mask = np.where(is_static, np.ones(2, dtype=np.float32), stance_mask)

        if not self.fixed_base:
            root_pos = (
                path_state["path_rot"].apply(self.default_pos) + path_state["path_pos"]
            )
            root_rot = path_state["path_rot"] * self.default_rot
            # Convert to MuJoCo qpos (wxyz format)
            root_quat_xyzw = root_rot.as_quat()  # xyzw
            root_quat = np.concatenate([root_quat_xyzw[3:], root_quat_xyzw[:3]])
            qpos = np.concatenate([root_pos, root_quat, qpos])

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
