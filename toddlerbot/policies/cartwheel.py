"""Cartwheel movement policy for dynamic acrobatic maneuvers.

This module implements a cartwheel policy that extends MJXPolicy to perform
dynamic cartwheel movements with phase-based motion generation.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.cartwheel_ref import CartwheelReference
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class CartwheelPolicy(MJXPolicy):
    """Cartwheel policy for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(name, robot, init_motor_pos, path, joystick, fixed_command)

        motion_ref = CartwheelReference(robot, self.control_dt)
        state_ref = motion_ref.get_default_state()
        state_ref = motion_ref.get_state_ref(0.0, np.zeros(3), state_ref)
        self.default_action = state_ref["motor_pos"][self.action_mask]
        self.ref_motor_pos = state_ref["motor_pos"].copy()

    def get_phase_signal(self, time_curr: float, num_frames: int = 215):
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)
        # Calculate phase based on total_idx
        phase = ((time_idx % num_frames) / num_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        return phase_signal
