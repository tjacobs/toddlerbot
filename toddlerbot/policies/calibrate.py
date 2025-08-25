"""Robot zero point calibration policy using torso pitch PID control.

This module implements a calibration policy that uses a PID controller to maintain
the robot's torso pitch at zero degrees for accurate zero-point calibration of motors.
"""

import os
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import yaml

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class CalibratePolicy(BasePolicy):
    """Policy for calibrating zero point with the robot's torso pitch."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        kp: float = 0.1,
        kd: float = 0.01,
        ki: float = 0.2,
    ):
        """Initializes the controller with specified parameters and robot configuration.

        Args:
            name (str): The name of the controller.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            kp (float, optional): Proportional gain for the controller. Defaults to 0.1.
            kd (float, optional): Derivative gain for the controller. Defaults to 0.01.
            ki (float, optional): Integral gain for the controller. Defaults to 0.2.
        """
        super().__init__(name, robot, init_motor_pos)

        leg_pitch_joint_names = [
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
        ]
        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in leg_pitch_joint_names
            ]
        )
        self.leg_pitch_joint_signs = np.array([-1, -1, 1, 1, 1, -1], dtype=np.float32)

        # PD controller parameters
        self.kp = kp
        self.kd = kd
        self.ki = ki

        # Initialize integral error
        self.integral_error = 0.0
        self.last_motor_pos = self.default_motor_pos.copy()
        self.is_real = None

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step to maintain the torso pitch at zero using a PD+I controller.

        Args:
            obs (Obs): The current observation containing state information such as time, Euler angles, and angular velocities.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and an array of motor target angles.
        """
        # Preparation phase
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        if self.is_real is None:
            self.is_real = "real" in sim.name

        self.last_motor_pos = obs.motor_pos.copy()

        # PD+I controller to maintain torso pitch at 0
        error = obs.rot.as_euler("xyz")[1]  # + 0.05  # cancels some backlash
        error_derivative = obs.ang_vel[1]

        # Update integral error (with a basic anti-windup mechanism)
        self.integral_error += error * self.control_dt
        self.integral_error = np.clip(self.integral_error, -10.0, 10.0)  # Anti-windup

        # PID controller output
        ctrl = (
            self.kp * error + self.ki * self.integral_error - self.kd * error_derivative
        )

        # Update joint positions based on the PID controller command
        joint_pos = self.default_joint_pos.copy()
        joint_pos[self.leg_pitch_joint_indicies] += self.leg_pitch_joint_signs * ctrl

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        return {}, motor_target

    def close(self, exp_folder_path: str):
        """Save calibrated motor zero positions to config file."""
        motor_config_path = os.path.join(
            "toddlerbot", "descriptions", self.robot.name, "motors.yml"
        )
        if os.path.exists(motor_config_path) and self.is_real:
            motor_config = yaml.safe_load(open(motor_config_path, "r"))

            motor_pos_delta = self.last_motor_pos - self.default_motor_pos
            motor_pos_delta[
                np.logical_and(motor_pos_delta > -0.005, motor_pos_delta < 0.005)
            ] = 0.0
            for motor_name, zero_pos in zip(
                self.robot.motor_ordering, self.robot.motor_zero_pos + motor_pos_delta
            ):
                motor_config["motors"][motor_name]["zero_pos"] = round(
                    float(zero_pos), 6
                )

            with open(motor_config_path, "w") as f:
                yaml.dump(
                    motor_config, f, indent=4, default_flow_style=False, sort_keys=False
                )
