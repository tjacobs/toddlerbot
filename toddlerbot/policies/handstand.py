"""Handstand policy for performing handstand maneuvers.

This module implements a handstand policy that positions the robot
into a handstand pose with arms extended upward.
"""

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action


class HandstandPolicy(BasePolicy):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the object with a name, a robot instance, and initial motor positions.

        Args:
            name (str): The name of the object.
            robot (Robot): An instance of the Robot class.
            init_motor_pos (npt.NDArray[np.float32]): An array of initial motor positions.
        """
        super().__init__(name, robot, init_motor_pos)

        self.is_prepared = False
        self.ref_motor_pos = self.default_motor_pos.copy()
        left_shoulder_pitch_idx = self.robot.motor_ordering.index("left_shoulder_pitch")
        right_shoulder_pitch_idx = self.robot.motor_ordering.index(
            "right_shoulder_pitch"
        )
        self.ref_motor_pos[self.arm_motor_indices] = 0.0
        self.ref_motor_pos[left_shoulder_pitch_idx] = -3.14
        self.ref_motor_pos[right_shoulder_pitch_idx] = 3.14

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Generates the next action based on the current observation and preparation phase.

        Args:
            obs (Obs): The current observation containing the time and other relevant data.
            is_real (bool, optional): Flag indicating if the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the next action as a numpy array. If the current time is within the preparation duration, the action is interpolated based on the preparation time and action; otherwise, the default motor position is returned.
        """
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 2.0
            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                self.init_motor_pos,
                self.ref_motor_pos,
                self.prep_duration,
                self.control_dt,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        return {}, self.ref_motor_pos
