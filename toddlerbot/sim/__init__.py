from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


@dataclass
class Obs:
    """Observation data structure for robot state information.

    This dataclass encapsulates all relevant sensor and state information
    from the robot simulation or real hardware, including motor states,
    IMU data, and joint positions/velocities.

    Attributes:
        time: Timestamp of the observation in seconds.
        motor_pos: Motor positions in radians as a float32 array.
        motor_vel: Motor velocities in rad/s as a float32 array.
        motor_acc: Optional motor accelerations in rad/s² as a float32 array.
        motor_cur: Optional motor currents in Amperes as a float32 array.
        motor_tor: Optional motor torques in N⋅m as a float32 array.
        pos: Optional 3D position [x, y, z] in meters as a float32 array.
        rot: Optional orientation as a scipy Rotation object.
        lin_vel: Optional linear velocity [vx, vy, vz] in m/s as a float32 array.
        ang_vel: Optional angular velocity [wx, wy, wz] in rad/s as a float32 array.
        joint_pos: Optional joint positions in radians as a float32 array.
        joint_vel: Optional joint velocities in rad/s as a float32 array.
    """

    time: float
    motor_pos: npt.NDArray[np.float32]
    motor_vel: npt.NDArray[np.float32]
    motor_acc: Optional[npt.NDArray[np.float32]] = None
    motor_cur: Optional[npt.NDArray[np.float32]] = None
    motor_tor: Optional[npt.NDArray[np.float32]] = None
    pos: Optional[npt.NDArray[np.float32]] = None
    rot: Optional[R] = None
    lin_vel: Optional[npt.NDArray[np.float32]] = None
    ang_vel: Optional[npt.NDArray[np.float32]] = None
    joint_pos: Optional[npt.NDArray[np.float32]] = None
    joint_vel: Optional[npt.NDArray[np.float32]] = None


class BaseSim(ABC):
    """Base class for simulation environments and real robot interfaces.

    This abstract base class defines the common interface for both simulation
    environments (like MuJoCo) and real robot hardware interfaces. It provides
    a unified API for motor control, observation collection, and simulation stepping.

    All concrete implementations must override the abstract methods to provide
    specific functionality for their respective environments.
    """

    @abstractmethod
    def __init__(self, name: str):
        """Initialize the simulation or robot interface.

        Args:
            name: Identifier name for this simulation/robot instance.
        """
        self.name = name

    @abstractmethod
    def set_motor_target(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Set target positions for all motors.

        Args:
            motor_angles: Target motor positions, either as a dictionary mapping
                motor names to angles, or as a numpy array of motor angles.
        """
        pass

    @abstractmethod
    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Set proportional gains for motor position control.

        Args:
            motor_kps: Dictionary mapping motor names to proportional gain values.
        """
        pass

    @abstractmethod
    def step(self):
        """Advance the simulation by one timestep or execute one control cycle.

        This method should update the simulation state or send control commands
        to the real robot hardware.
        """
        pass

    @abstractmethod
    def get_observation(self) -> Obs:
        """Retrieve the current observation from the simulation or robot.

        Returns:
            An Obs dataclass containing the current robot state information
            including motor positions, velocities, and sensor data.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up and close the simulation or robot connection.

        This method should properly release resources, close connections,
        and perform any necessary cleanup operations.
        """
        pass
