"""Walking locomotion policy with phase-based gait control.

This module implements a walking policy that generates coordinated
bipedal locomotion using cyclic phase signals and command interpretation.
"""

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim import BaseSim, Obs


class WalkPolicy(MJXPolicy):
    """Walking policy for the toddlerbot robot."""

    def get_phase_signal(self, time_curr: float):
        """Calculate the phase signal as a 2D vector for a given time.

        Args:
            time_curr (float): The current time for which to calculate the phase signal.

        Returns:
            np.ndarray: A 2D vector containing the sine and cosine components of the phase signal, with dtype np.float32.
        """
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.env_cfg["action"]["cycle_time"]),
                np.cos(2 * np.pi * time_curr / self.env_cfg["action"]["cycle_time"]),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_command(
        self, obs: Obs, control_inputs: Dict[str, float]
    ) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs for walking.

        Args:
            control_inputs (Dict[str, float]): A dictionary containing control inputs with keys 'walk_x', 'walk_y', and 'walk_turn'.

        Returns:
            npt.NDArray[np.float32]: A numpy array representing the command, with the first five elements as zeros and the remaining elements scaled by the command discount factor.
        """
        if len(control_inputs) == 0:
            command = self.fixed_command.copy()
        else:
            command = np.zeros(self.num_commands, dtype=np.float32)
            command[5:] = np.array(
                [
                    control_inputs["walk_x"],
                    control_inputs["walk_y"],
                    control_inputs["walk_turn"],
                ]
            )

        self.target_torso_yaw += command[-1] * self.control_dt

        torso_yaw = obs.rot.as_euler("xyz")[2]
        command[-1] += self.yaw_corr_gain * (self.target_torso_yaw - torso_yaw)

        # Clamp yaw rate
        command[-1] = np.clip(
            command[-1], self.command_range[-1][0], self.command_range[-1][1]
        )

        # print(f"walk_command: {command[5:]}")
        return command

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step based on the observed state and updates the standing status.

        Args:
            obs (Obs): The current observation of the system state.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs as a dictionary and the motor target as a NumPy array.
        """
        control_inputs, motor_target = super().step(obs, sim)

        if len(self.command_list) >= int(1 / self.control_dt):
            last_commands = self.command_list[-int(1 / self.control_dt) :]
            all_zeros = all(np.all(command == 0) for command in last_commands)
            self.is_standing = all_zeros and abs(self.phase_signal[1]) > 1 - 1e-6
        else:
            self.is_standing = False

        return control_inputs, motor_target
