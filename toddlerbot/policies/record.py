"""Motion recording policy for data collection during teleoperation.

This module implements a recording policy that captures robot motion data
while selectively disabling certain motors for natural movement recording.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode


class RecordPolicy(BalancePDPolicy):
    """Policy for recording the robot's motion data."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        cameras: Optional[List[Camera]] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes an instance of the class with specified parameters.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot associated with this instance.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the motors.
            joystick (Optional[Joystick]): The joystick for controlling the robot, if any.
            cameras (Optional[List[Camera]]): List of cameras attached to the robot, if any.
            zmq_receiver (Optional[ZMQNode]): ZMQ node for receiving data, if any.
            zmq_sender (Optional[ZMQNode]): ZMQ node for sending data, if any.
            ip (str): IP address for network communication.
            fixed_command (Optional[npt.NDArray[np.float32]]): A fixed command to be used, if any.
        """
        super().__init__(
            name,
            robot,
            init_motor_pos,
            joystick,
            cameras,
            zmq_receiver,
            zmq_sender,
            ip,
            fixed_command,
        )

        self.disable_motor_indices = np.concatenate([self.leg_motor_indices])

        self.is_prepared = False
        self.is_running = False

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        """Retrieves the current position of the arm motor.

        If the arm motor position is not set, it returns the default motor position for the arm.

        Args:
            obs (Obs): The observation object containing relevant data.

        Returns:
            npt.NDArray[np.float32]: The current or default position of the arm motor.
        """
        if self.arm_motor_pos is None:
            return self.default_motor_pos[self.arm_motor_indices]
        else:
            return self.arm_motor_pos

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the control process, adjusting motor targets based on observation and preparation duration.

        Args:
            obs (Obs): The current observation containing time and motor positions.
            is_real (bool, optional): Flag indicating if the step is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs and the adjusted motor target positions.
        """
        control_inputs, motor_target = super().step(obs, sim)

        if obs.time >= self.prep_duration:
            if not self.is_running:
                self.is_running = True
                if isinstance(sim, RealWorld):
                    sim.dynamixel_controller.disable_motors(self.disable_motor_indices)

            motor_target[self.disable_motor_indices] = obs.motor_pos[
                self.disable_motor_indices
            ]

        return control_inputs, motor_target
