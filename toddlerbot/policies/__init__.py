"""Policy framework for ToddlerBot robot control and behavior.

This module provides the base infrastructure for implementing robot control policies,
including support for various behavioral patterns such as walking, standing,
manipulation, and teleoperation modes.

The policy framework supports:
- Unified interface for different control strategies
- Smooth transitions between motor positions
- Configurable preparation phases for policy initialization
- Integration with both simulation and real robot hardware

All policies inherit from BasePolicy and implement domain-specific
control logic while maintaining consistent interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_action_traj
from toddlerbot.utils.misc_utils import snake2camel


class BasePolicy(ABC):
    """Abstract base class for all robot control policies.

    This class defines the common interface and shared functionality for robot
    control policies. It handles initialization, motor group indexing,
    preparation trajectories, and provides a consistent framework for
    implementing different behavioral patterns.

    All concrete policy implementations must inherit from this class and
    implement the abstract step() method to define their specific control logic.
    """

    @abstractmethod
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,
        prep_duration: float = 2.0,
        n_steps_total: float = float("inf"),
    ):
        """Initialize the policy with robot configuration and control parameters.

        Args:
            name: Identifier name for this policy instance.
            robot: Robot configuration object containing joint and motor specifications.
            init_motor_pos: Initial motor positions in radians as a float32 array.
            control_dt: Control loop timestep in seconds. Defaults to 0.02.
            prep_duration: Duration for preparation phase when transitioning from
                current to default positions. Defaults to 2.0.
            n_steps_total: Maximum number of control steps to execute. Defaults to
                infinity for continuous operation.
        """
        self.name = name
        self.robot = robot
        self.init_motor_pos = init_motor_pos
        self.control_dt = control_dt
        self.prep_duration = prep_duration
        self.n_steps_total = n_steps_total

        self.header_name = snake2camel(name)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.motor_limits = np.array(
            [robot.motor_limits[name] for name in robot.motor_ordering]
        )
        indices = np.arange(robot.nu)
        motor_groups = np.array(robot.motor_groups)
        joint_groups = np.array(robot.joint_groups)
        self.leg_motor_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_motor_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_motor_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_motor_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = get_action_traj(
            0.0,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            self.control_dt,
        )

    def reset(self):
        """Reset the policy state for a new episode or run.

        This method is called to reinitialize the policy state when starting
        a new control episode. Subclasses can override this method to implement
        specific reset behavior.
        """
        pass

    @abstractmethod
    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Execute one control step of the policy.

        Args:
            obs: Current observation containing robot state information.
            sim: Simulation or robot interface for sending control commands.

        Returns:
            A tuple containing:
            - Dictionary mapping motor names to target positions
            - Array of motor positions for all motors
        """
        pass

    def close(self, exp_folder_path: str = ""):
        """Clean up resources and save any final data.

        Args:
            exp_folder_path: Optional path to save experiment data or logs.
                Defaults to empty string.
        """
        pass
