"""Abstract base class for motion references in toddlerbot.

Defines the interface for generating motion references and provides common functionality
for robot motion control including path integration and state management.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import mujoco
import numpy

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, R, inplace_update
from toddlerbot.utils.array_utils import array_lib as np

# from toddlerbot.utils.misc_utils import profile


@dataclass
class Motion:
    """Data structure for storing motion reference data."""

    time: ArrayType
    qpos: ArrayType
    body_pos: ArrayType
    body_quat: ArrayType
    site_pos: ArrayType
    site_quat: ArrayType
    action: Optional[ArrayType] = None
    contact: Optional[ArrayType] = None


class MotionReference(ABC):
    """Abstract class for generating motion references for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        motion_type: str,
        robot: Robot,
        dt: float,
        fixed_base: bool = False,
    ):
        """Initializes the motion controller for a robot with specified parameters.

        Args:
            name (str): The name of the motion controller.
            motion_type (str): The type of motion to be controlled (e.g., 'walking', 'running').
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for the control loop.
            com_kp (List[float], optional): The proportional gain for the center of mass control. Defaults to [1.0, 1.0].
        """
        self.name = name
        self.motion_type = motion_type
        self.robot = robot
        self.dt = dt
        self.fixed_base = fixed_base
        self.use_jax = os.environ.get("USE_JAX", "false") == "true"

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )

        indices = np.arange(robot.nu)
        motor_groups = numpy.array(robot.motor_groups)
        joint_groups = numpy.array(robot.joint_groups)
        self.leg_motor_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_motor_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_motor_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_motor_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

        description_dir = os.path.join("toddlerbot", "descriptions")
        xml_suffix = ""
        if self.fixed_base:
            xml_suffix += "_fixed"
        xml_path = os.path.join(
            description_dir, self.robot.name, f"scene{xml_suffix}.xml"
        )
        model = mujoco.MjModel.from_xml_path(xml_path)

        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        # self.renderer = mujoco.Renderer(model)
        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.default_pos = self.default_qpos[:3]
        default_quat = self.default_qpos[3:7]
        default_quat_xyzw = np.concatenate((default_quat[1:], default_quat[:1]), axis=0)
        self.default_rot = R.from_quat(default_quat_xyzw)

        self.mj_motor_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        # Account for the free joint
        if not self.fixed_base:
            self.mj_motor_indices -= 1
            self.mj_joint_indices -= 1

        if self.robot.has_gripper:
            mj_gripper_indices = np.array(
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in self.robot.gripper_dof_names
                ]
            )
            if not self.fixed_base:
                mj_gripper_indices -= 1

            self.mj_gripper_mask = np.zeros(model.nv - self.qd_start_idx, dtype=bool)
            self.mj_gripper_mask = inplace_update(
                self.mj_gripper_mask, mj_gripper_indices, True
            )

        self.left_hand_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_center"
        )
        self.right_hand_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_center"
        )
        self.left_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_center"
        )
        self.right_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_center"
        )

        # Load the balance dataset
        data_path = os.path.join("motion", "arm_pose_dataset.lz4")
        data_dict = joblib.load(data_path)
        time_arr = data_dict["time"]
        motor_pos_arr = data_dict["motor_pos"]
        self.arm_time_ref = np.array(time_arr - time_arr[0], dtype=np.float32)
        if motor_pos_arr.shape[1] < len(self.arm_motor_indices):
            gripper_padding = np.zeros((motor_pos_arr.shape[0], 2), dtype=np.float32)
            motor_pos_arr = np.concatenate([motor_pos_arr, gripper_padding], axis=1)

        self.arm_joint_pos_ref = np.array(
            [
                self.robot.arm_fk(arm_motor_pos)
                for arm_motor_pos in motor_pos_arr[:, : len(self.arm_motor_indices)]
            ],
            dtype=np.float32,
        )
        self.arm_ref_size = len(self.arm_time_ref)

    def get_default_state(self) -> Dict[str, ArrayType]:
        """Returns the default state of the robot, including position, orientation, and velocities.

        This method initializes the robot's state with default values for position, orientation (as a quaternion), linear velocity, angular velocity, motor positions, joint positions, and stance mask.

        Returns:
            Dict[str, ArrayType]: A dictionary containing the default state of the robot.
        """
        return {
            "path_pos": np.zeros(3, dtype=np.float32),
            "path_rot": R.identity(),
            "lin_vel": np.zeros(3, dtype=np.float32),
            "ang_vel": np.zeros(3, dtype=np.float32),
            "motor_pos": self.default_motor_pos.copy(),
            "joint_pos": self.default_joint_pos.copy(),
            "qpos": self.default_qpos[7:].copy(),
            "stance_mask": np.ones(2, dtype=np.float32),
        }

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        """Calculate the phase signal at a given time.

        Args:
            time_curr (float | ArrayType): The current time or an array of time values.

        Returns:
            ArrayType: An array containing the phase signal, initialized to zeros.
        """
        return np.zeros(1, dtype=np.float32)

    @abstractmethod
    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        pass

    def integrate_path_state(
        self, command: ArrayType, last_state: Dict[str, ArrayType]
    ) -> Dict[str, ArrayType]:
        """Integrates the current state of a path with a given command to compute the next state.

        This function calculates the new position and orientation of an object by integrating
        its current state with a command that specifies linear and angular velocities. The
        orientation is updated using quaternion multiplication to apply roll, pitch, and yaw
        rotations.

        Args:
            last_state (ArrayType): The current state of the object, including position,
                orientation (as a quaternion), linear velocity, and angular velocity.
            command (ArrayType): The command input specifying desired linear and angular velocities.

        Returns:
            ArrayType: The updated state of the object, including new position, orientation,
            linear velocity, and angular velocity.
        """
        lin_vel, ang_vel = self.get_vel(command)

        rotvec = ang_vel * self.dt
        delta_rot = R.from_rotvec(rotvec)

        curr_rot = last_state["path_rot"]  # [x, y, z, w] for scipy
        path_rot = curr_rot * delta_rot

        rotated_lin_vel = path_rot.apply(lin_vel)
        path_pos = last_state["path_pos"] + rotated_lin_vel * self.dt

        return {
            "path_pos": path_pos,
            "path_rot": path_rot,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
        }

    @abstractmethod
    def get_state_ref(
        self,
        time_curr: float | ArrayType,
        command: ArrayType,
        last_state: Dict[str, ArrayType],
        init_idx: int = 0,
    ) -> Dict[str, ArrayType]:
        pass
