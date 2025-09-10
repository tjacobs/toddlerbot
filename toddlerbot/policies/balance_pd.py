"""PD control policy for robot balancing and basic manipulation tasks.

This module provides a balance control policy using proportional-derivative (PD) controllers
for center of mass positioning, torso orientation control, and basic manipulation capabilities.
"""

import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action

# from toddlerbot.utils.misc_utils import profile


class BalancePDPolicy(BasePolicy):
    """Policy for balancing the robot using a PD controller."""

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
        use_torso_pd: bool = True,
    ):
        """Initializes the control system for a robot, setting up various components such as joystick, cameras, and ZeroMQ nodes for communication.

        Args:
            name (str): The name of the control system.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the robot's motors.
            joystick (Optional[Joystick], optional): Joystick for manual control. Defaults to None.
            cameras (Optional[List[Camera]], optional): List of camera objects for visual input. Defaults to None.
            zmq_receiver (Optional[ZMQNode], optional): ZeroMQ node for receiving data. Defaults to None.
            zmq_sender (Optional[ZMQNode], optional): ZeroMQ node for sending data. Defaults to None.
            ip (str, optional): IP address for ZeroMQ communication. Defaults to an empty string.
            fixed_command (Optional[npt.NDArray[np.float32]], optional): Fixed command array for the robot. Defaults to None.
            use_torso_pd (bool, optional): Flag to use proportional-derivative control for the torso. Defaults to True.
        """
        super().__init__(name, robot, init_motor_pos)

        self.command_range = np.array(
            [
                [-1.5, 1.5],
                [-1.5, 1.5],
                [0.0, 0.5],
                [-0.3, 0.3],
                [-1.5, 1.5],
                [-0.1, 0.1],
            ],
            dtype=np.float32,
        )
        self.num_commands = len(self.command_range)

        self.zero_command = np.zeros(self.num_commands, dtype=np.float32)
        self.fixed_command = (
            self.zero_command if fixed_command is None else fixed_command
        )
        self.use_torso_pd = use_torso_pd

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        sys_name = platform.system()
        self.zmq_receiver = None
        if zmq_receiver is not None:
            self.zmq_receiver = zmq_receiver
        elif sys_name != "Darwin":
            self.zmq_receiver = ZMQNode(type="receiver")

        self.zmq_sender = None
        if zmq_sender is not None:
            self.zmq_sender = zmq_sender
        elif sys_name != "Darwin":
            self.zmq_sender = ZMQNode(type="sender", ip=ip)

        self.left_eye = None
        self.right_eye = None
        if cameras is not None:
            self.left_eye = cameras[0]
            self.right_eye = cameras[1]
        elif sys_name != "Darwin":
            try:
                self.left_eye = Camera("left")
                self.right_eye = Camera("right")
            except Exception:
                pass

        self.capture_rgb = False

        self.msg = None
        self.is_running = False
        self.is_button_pressed = False
        self.is_ended = False
        self.last_control_inputs: Dict[str, float] = {}
        self.step_curr = 0

        self.camera_frame: npt.NDArray[np.uint8] | None = None
        self.camera_time_list: List[float] = []
        self.camera_frame_list: List[npt.NDArray[np.uint8]] = []

        self.arm_motor_pos = None
        self.last_arm_motor_pos = None
        self.last_gripper_pos = np.zeros(2, dtype=np.float32)
        # Limits the max delta for arm and gripper movements
        self.arm_delta_max = 0.2
        self.gripper_delta_max = 0.5
        self.last_joint_target = self.default_joint_pos.copy()

        self.neck_motor_pos = None
        self.last_neck_motor_pos = None
        self.neck_delta_max = 0.1

        # These are for the center of mass (CoM) PD control
        self.hip_to_knee_z = robot.config["robot"]["hip_to_knee_z"]
        self.knee_to_ank_z = robot.config["robot"]["knee_to_ankle_z"]
        self.hip_to_ank_pitch = np.array(
            [0, 0, robot.config["robot"]["hip_to_ankle_pitch_z"]], dtype=np.float32
        )
        self.hip_to_ank_roll = np.array(
            [0, 0, robot.config["robot"]["hip_to_ankle_roll_z"]], dtype=np.float32
        )
        knee_max = np.max(
            np.abs(np.array(robot.joint_limits["left_knee"], dtype=np.float32))
        )
        self.com_z_limits = np.array(
            [self.com_fk(knee_max)[2] + 0.01, 0.0],
            dtype=np.float32,
        )
        self.com_kp = np.array([1.0, 1.0], dtype=np.float32)
        self.desired_com = np.zeros(2, dtype=np.float32)
        self.com_x_init = robot.config["robot"]["com_x"]
        self.last_com_z = robot.config["kinematics"]["home_pos_z_delta"]

        self.sim = MuJoCoSim(robot)
        self.sim.set_qpos(self.sim.model.keyframe("home").qpos)
        self.sim.forward()
        self.left_foot_t_init = self.sim.get_body_transform(
            "left_" + self.sim.robot.foot_name
        )
        self.right_foot_t_init = self.sim.get_body_transform(
            "right_" + self.sim.robot.foot_name
        )

        # These are for torso roll and pitch PD control
        self.desired_torso_pitch = -0.2  # -0.7 for the payload test
        self.desired_torso_roll = 0.0
        self.last_torso_pitch = 0.0
        self.last_torso_roll = 0.0
        self.torso_roll_kp = 0.2
        self.torso_roll_kd = 0.0
        self.torso_pitch_kp = 0.2
        self.torso_pitch_kd = 0.01

        self.left_sho_pitch_idx = robot.motor_ordering.index("left_shoulder_pitch")
        self.right_sho_pitch_idx = robot.motor_ordering.index("right_shoulder_pitch")
        self.left_sho_roll_idx = robot.motor_ordering.index("left_shoulder_roll")
        self.right_sho_roll_idx = robot.motor_ordering.index("right_shoulder_roll")

        joint_signs = (np.sign(self.default_joint_pos) >= 0).astype(np.float32) * 2 - 1
        left_hip_roll_idx = robot.motor_ordering.index("left_hip_roll")
        right_hip_roll_idx = robot.motor_ordering.index("right_hip_roll")
        left_ank_roll_idx = robot.motor_ordering.index("left_ankle_roll")
        right_ank_roll_idx = robot.motor_ordering.index("right_ankle_roll")
        joint_signs[left_hip_roll_idx] = 1.0
        joint_signs[right_hip_roll_idx] = 1.0
        joint_signs[left_ank_roll_idx] = 1.0
        joint_signs[right_ank_roll_idx] = 1.0
        self.leg_joint_signs = joint_signs[self.leg_joint_indices]

        self.is_prepared = False
        self.prep_duration = 7.0
        self.time_start = self.prep_duration

        self.is_ready = False
        self.manip_duration = 2.0
        self.manip_motor_pos = self.default_motor_pos.copy()

    def get_command(
        self, control_inputs: Dict[str, float]
    ) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs for various tasks.

        This function processes a dictionary of control inputs, mapping each task to a corresponding command value within a predefined range. It also manages the state of a button press to toggle logging.

        Args:
            control_inputs (Dict[str, float]): A dictionary where keys are task names and values are the corresponding input values.

        Returns:
            npt.NDArray[np.float32]: An array of command values, each corresponding to a task, with values adjusted according to the input and predefined command ranges.
        """
        command = np.zeros(len(self.command_range), dtype=np.float32)
        for task, input in control_inputs.items():
            if task in self.name:
                if abs(input) > 0.5:
                    # Button is pressed
                    if not self.is_button_pressed:
                        self.is_button_pressed = True  # Mark the button as pressed
                        self.is_running = not self.is_running  # Toggle logging

                        if not self.is_running:
                            self.is_ended = True

                        print(
                            f"\nLogging is now {'enabled' if self.is_running else 'disabled'}."
                        )
                else:
                    # Button is released
                    self.is_button_pressed = False  # Reset button pressed state

            elif task == "look_left" and input > 0:
                command[0] = input * self.command_range[0][1]
            elif task == "look_right" and input > 0:
                command[0] = input * self.command_range[0][0]
            elif task == "look_up" and input > 0:
                command[1] = input * self.command_range[1][1]
            elif task == "look_down" and input > 0:
                command[1] = input * self.command_range[1][0]
            elif task == "lean_left" and input > 0:
                command[3] = input * self.command_range[3][0]
            elif task == "lean_right" and input > 0:
                command[3] = input * self.command_range[3][1]
            elif task == "twist_left" and input > 0:
                command[4] = input * self.command_range[4][0]
            elif task == "twist_right" and input > 0:
                command[4] = input * self.command_range[4][1]
            elif task == "squat":
                command[5] = np.interp(
                    input,
                    [-1, 0, 1],
                    [self.command_range[5][1], 0.0, self.command_range[5][0]],
                )

        return command

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        """Retrieve the positions of the arm motors from the observation data.

        Args:
            obs (Obs): The observation data containing motor positions.

        Returns:
            npt.NDArray[np.float32]: An array of the arm motor positions.
        """
        return self.manip_motor_pos[self.arm_motor_indices]

    def get_neck_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        """Retrieve the positions of the arm motors from the observation data.

        Args:
            obs (Obs): The observation data containing motor positions.

        Returns:
            npt.NDArray[np.float32]: An array of the arm motor positions.
        """
        return self.manip_motor_pos[self.neck_motor_indices]

    def com_fk(self, knee_angle, hip_pitch_angle=None, hip_roll_angle=None):
        """Calculates the center of mass (CoM) position of a leg segment based on joint angles.

        Args:
            knee_angle (float | ArrayType): The angle of the knee joint in radians.
            hip_pitch_angle (Optional[float | ArrayType]): The angle of the hip pitch joint in radians. Defaults to None.
            hip_roll_angle (Optional[float | ArrayType]): The angle of the hip roll joint in radians. Defaults to None.

        Returns:
            ArrayType: A 3D vector representing the CoM position in Cartesian coordinates.
        """
        hip_to_ank = np.sqrt(
            self.hip_to_knee_z**2
            + self.knee_to_ank_z**2
            - 2 * self.hip_to_knee_z * self.knee_to_ank_z * np.cos(np.pi - knee_angle)
        )

        if hip_pitch_angle is None:
            alpha = 0.0
        else:
            alpha = (
                np.arcsin(self.knee_to_ank_z / hip_to_ank * np.sin(knee_angle))
                + hip_pitch_angle
            )

        if hip_roll_angle is None:
            hip_roll_angle = 0.0

        com_x = hip_to_ank * np.sin(alpha) * np.cos(hip_roll_angle)
        com_y = hip_to_ank * np.cos(alpha) * np.sin(hip_roll_angle)
        com_z = (
            hip_to_ank * np.cos(alpha) * np.cos(hip_roll_angle)
            - self.hip_to_ank_pitch[2]
        )
        return np.array([com_x, com_y, com_z], dtype=np.float32)

    def com_ik(self, com_z, com_x=None, com_y=None):
        """Calculates the inverse kinematics for the center of mass (COM) of a bipedal robot leg.

        This function computes the joint angles required to position the robot's leg such that the center of mass is at the specified coordinates. It uses the lengths of the leg segments and default positions to determine the necessary joint angles.

        Args:
            com_z (float or ArrayType): The z-coordinate of the center of mass.
            com_x (Optional[float or ArrayType]): The x-coordinate of the center of mass. Defaults to 0.0 if not provided.
            com_y (Optional[float or ArrayType]): The y-coordinate of the center of mass. Defaults to 0.0 if not provided.

        Returns:
            ArrayType: An array of joint angles for the leg, including hip pitch, hip roll, knee, and ankle pitch.
        """
        if com_x is None:
            com_x = 0.0
        if com_y is None:
            com_y = 0.0

        com_x -= self.com_x_init

        hip_to_ank_pitch_target = self.hip_to_ank_pitch + np.array(
            [com_x, 0, com_z], dtype=np.float32
        )
        hip_to_ank_roll_target = self.hip_to_ank_roll + np.array(
            [0, com_y, com_z], dtype=np.float32
        )

        knee_cos = (
            self.hip_to_knee_z**2
            + self.knee_to_ank_z**2
            - np.linalg.norm(hip_to_ank_pitch_target) ** 2
        ) / (2 * self.hip_to_knee_z * self.knee_to_ank_z)
        knee_cos = np.clip(knee_cos, -1.0, 1.0)
        knee = np.abs(np.pi - np.arccos(knee_cos))

        ank_pitch = np.arctan2(
            np.sin(knee), np.cos(knee) + self.knee_to_ank_z / self.hip_to_knee_z
        ) + np.arctan2(hip_to_ank_pitch_target[0], hip_to_ank_pitch_target[2])
        hip_pitch = knee - ank_pitch

        hip_roll_cos = np.dot(hip_to_ank_roll_target, np.array([0, 0, 1])) / (
            np.linalg.norm(hip_to_ank_roll_target)
        )
        hip_roll_cos = np.clip(hip_roll_cos, -1.0, 1.0)
        hip_roll = np.arccos(hip_roll_cos) * np.sign(hip_to_ank_roll_target[1])

        leg_joint_pos = (
            np.array(
                [hip_pitch, hip_roll, 0.0, knee, hip_roll, ank_pitch] * 2,
                dtype=np.float32,
            )
            * self.leg_joint_signs
        )

        return leg_joint_pos

    def get_motor_target(
        self, arm_joint_pos: npt.NDArray[np.float32], command: npt.NDArray[np.float32], neck_joint_pos: npt.NDArray[np.float32]=None
    ):
        joint_target = self.default_joint_pos.copy()
        if neck_joint_pos is None:
                neck_joint_pos = np.clip(
                    self.last_joint_target[self.neck_joint_indices]
                + self.control_dt * command[:2],
                self.robot.neck_joint_limits[0],
                self.robot.neck_joint_limits[1],
            )
        else:
            neck_joint_pos = np.clip(
                neck_joint_pos,
                self.robot.neck_joint_limits[0],
                self.robot.neck_joint_limits[1],
            )
        joint_target[self.neck_joint_indices] = neck_joint_pos

        waist_joint_pos = np.clip(
            self.last_joint_target[self.waist_joint_indices]
            + self.control_dt * command[3:5],
            self.robot.waist_joint_limits[0],
            self.robot.waist_joint_limits[1],
        )
        joint_target[self.waist_joint_indices] = waist_joint_pos

        joint_target[self.arm_joint_indices] = arm_joint_pos

        com_z_target = np.clip(
            self.last_com_z + self.control_dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_joint_pos = self.com_ik(com_z_target)
        joint_target[self.leg_joint_indices] = leg_joint_pos

        self.sim.set_joint_angles(joint_target)
        self.sim.forward()

        # put feet on the ground
        left_foot_t_curr = self.sim.get_body_transform(
            "left_" + self.sim.robot.foot_name
        )
        right_foot_t_curr = self.sim.get_body_transform(
            "right_" + self.sim.robot.foot_name
        )
        torso_t_curr = self.sim.get_body_transform("torso")
        # Select the foot with the smaller z-coordinate
        if left_foot_t_curr[2, 3] < right_foot_t_curr[2, 3]:
            aligned_torso_t = (
                self.left_foot_t_init @ np.linalg.inv(left_foot_t_curr) @ torso_t_curr
            )
        else:
            aligned_torso_t = (
                self.right_foot_t_init @ np.linalg.inv(right_foot_t_curr) @ torso_t_curr
            )

        # Update the simulation with the new torso position and orientation
        self.sim.data.qpos[:3] = aligned_torso_t[:3, 3]
        self.sim.data.qpos[3:7] = R.from_matrix(aligned_torso_t[:3, :3]).as_quat(
            scalar_first=True
        )
        self.sim.forward()

        com_pos = np.array(self.sim.data.subtree_com[0], dtype=np.float32)
        # PD controller on CoM position
        com_pos_error = self.desired_com[:2] - com_pos[:2]
        com_ctrl = self.com_kp * com_pos_error

        leg_joint_pos = self.com_ik(com_z_target, com_ctrl[0], com_ctrl[1])
        joint_target[self.leg_joint_indices] = leg_joint_pos

        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_target))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        self.last_com_z = com_z_target
        self.last_joint_target = joint_target

        return motor_target

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Processes an observation to compute the next action and control inputs for a robotic system.

        Args:
            obs (Obs): The current observation containing sensor data and time information.
            is_real (bool, optional): Indicates if the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs and the target motor positions for the next step.
        """
        if not self.is_prepared:
            self.is_prepared = True
            is_real = "real" in sim.name
            if not is_real:
                self.prep_duration -= 5.0
                self.time_start = self.prep_duration

            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                self.control_dt,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action


        msg = None
        if self.msg is not None:
            msg = self.msg
        elif self.zmq_receiver is not None:
            msg = self.zmq_receiver.get_msg()
        # print(f"msg: {msg}")

        if msg is not None:
            # print(f"latency: {abs(time.time() - msg.time) * 1000:.2f} ms")
            if abs(time.time()- msg.time) < 1:
                if msg.action is None:
                    self.arm_motor_pos = self.last_arm_motor_pos
                    self.neck_motor_pos = self.last_neck_motor_pos
                elif hasattr(self, "task") and self.task == "teleop_vr":
                    self.arm_motor_pos = msg.action[2:]
                    self.neck_motor_pos = msg.action[:2]
                else:
                    self.arm_motor_pos = msg.action
                # self.arm_motor_pos = msg.action
                if (
                    self.last_arm_motor_pos is not None
                    and self.arm_motor_pos is not None
                ):
                    self.arm_motor_pos = np.clip(
                        self.arm_motor_pos,
                        self.last_arm_motor_pos - self.arm_delta_max,
                        self.last_arm_motor_pos + self.arm_delta_max,
                    )
                self.last_arm_motor_pos = self.arm_motor_pos

                if (
                    self.last_neck_motor_pos is not None
                    and self.neck_motor_pos is not None
                ):
                    self.neck_motor_pos = np.clip(
                        (self.neck_motor_pos + self.last_neck_motor_pos) / 2,
                        self.last_neck_motor_pos - self.neck_delta_max,
                        self.last_neck_motor_pos + self.neck_delta_max,
                    )

                if (
                    self.robot.has_gripper
                    and self.arm_motor_pos is not None
                    and msg.fsr is not None
                ):
                    gripper_pos = msg.fsr / 100 * self.motor_limits[-2:, 1]
                    gripper_pos = np.clip(
                        gripper_pos,
                        self.last_gripper_pos - self.gripper_delta_max,
                        self.last_gripper_pos + self.gripper_delta_max,
                    )
                    self.arm_motor_pos = np.concatenate(
                        [self.arm_motor_pos, gripper_pos]
                    )
                    self.last_gripper_pos = gripper_pos

                if self.arm_motor_pos is not None:
                    self.arm_motor_pos = np.clip(
                        self.arm_motor_pos,
                        self.motor_limits[self.arm_motor_indices, 0],
                        self.motor_limits[self.arm_motor_indices, 1],
                    )

                if self.neck_motor_pos is not None:
                    self.neck_motor_pos = np.clip(
                        self.neck_motor_pos,
                        self.motor_limits[self.neck_motor_indices, 0],
                        self.motor_limits[self.neck_motor_indices, 1],
                    )
            else:
                print("\nstale message received, discarding")

        if self.left_eye is not None and self.capture_rgb:
            jpeg_frame, self.camera_frame = self.left_eye.get_jpeg()
            assert self.camera_frame is not None
            self.camera_time_list.append(time.time())
            self.camera_frame_list.append(self.camera_frame)
        else:
            jpeg_frame = None

        if self.zmq_sender is not None:
            send_msg = ZMQMessage(time=time.time(), camera_frame=jpeg_frame)
            self.zmq_sender.send_msg(send_msg)

        control_inputs = self.last_control_inputs
        if self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()
        elif msg is not None and msg.control_inputs is not None:
            control_inputs = msg.control_inputs

        self.last_control_inputs = control_inputs

        if control_inputs is None:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)

        arm_motor_pos = self.get_arm_motor_pos(obs)
        arm_joint_pos = self.robot.arm_fk(arm_motor_pos)

        neck_motor_pos = self.get_neck_motor_pos(obs)
        neck_joint_pos = self.robot.neck_fk(neck_motor_pos)

        motor_target = self.get_motor_target(arm_joint_pos, command, neck_joint_pos)

        if self.use_torso_pd:
            current_roll, current_pitch, _ = obs.rot.as_euler("xyz")
            roll_error = self.desired_torso_roll - current_roll
            roll_vel = (current_roll - self.last_torso_roll) / self.control_dt
            pitch_error = self.desired_torso_pitch - current_pitch
            pitch_vel = (current_pitch - self.last_torso_pitch) / self.control_dt

            roll_pd_output = (
                self.torso_roll_kp * roll_error - self.torso_roll_kd * roll_vel
            )
            pitch_pd_output = (
                self.torso_pitch_kp * pitch_error - self.torso_roll_kd * pitch_vel
            )
            pd_output = np.array([roll_pd_output, pitch_pd_output, 0], dtype=np.float32)

            # Apply PD control based on torso pitch angle
            waist_roll, waist_yaw = self.robot.waist_fk(
                obs.motor_pos[self.waist_motor_indices]
            )
            waist_rot = R.from_euler("xyz", [waist_roll, 0.0, waist_yaw])
            pd_output_rotated = waist_rot.apply(pd_output)

            # print(f"waist_roll: {waist_roll:.2f}, waist_yaw: {waist_yaw:.2f}")
            # print(f"pd_output_rotated: {pd_output_rotated}")

            motor_target_delta = np.array(
                [
                    pd_output_rotated[1],
                    pd_output_rotated[0],
                    0.0,
                    0.0,
                    pd_output_rotated[0],
                    0.0,
                ]
                * 2,
                dtype=np.float32,
            )
            motor_target[self.leg_motor_indices] += (
                motor_target_delta * self.leg_joint_signs
            )

            self.last_torso_roll = current_roll
            self.last_torso_pitch = current_pitch

        # Override motor target with reference motion or teleop motion
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        if not self.is_ready:
            self.is_ready = True

            if hasattr(self, "task"): 
                if self.task == "hug":
                    manip_motor_pos_1 = self.manip_motor_pos.copy()
                    manip_motor_pos_1[self.left_sho_pitch_idx] = self.default_motor_pos[
                        self.left_sho_pitch_idx
                    ]
                    manip_motor_pos_1[self.right_sho_pitch_idx] = self.default_motor_pos[
                        self.right_sho_pitch_idx
                    ]
                    manip_motor_pos_1[self.left_sho_roll_idx] = -1.4
                    manip_motor_pos_1[self.right_sho_roll_idx] = -1.4

                    manip_time_1, manip_action_1 = get_action_traj(
                        0.0,
                        self.default_motor_pos,
                        manip_motor_pos_1,
                        self.manip_duration / 2,
                        self.control_dt,
                    )
                    manip_time_2, manip_action_2 = get_action_traj(
                        manip_time_1[-1] + self.control_dt,
                        manip_motor_pos_1,
                        self.manip_motor_pos,
                        self.manip_duration / 2,
                        self.control_dt,
                    )
                    self.manip_time = np.concatenate([manip_time_1, manip_time_2])
                    self.manip_action = np.concatenate([manip_action_1, manip_action_2])
                elif self.task == "teleop_vr":
                    msg = None
                    if self.msg is not None:
                        msg = self.msg
                    elif self.zmq_receiver is not None:
                        msg = self.zmq_receiver.get_msg()
                    self.manip_motor_pos = motor_target.copy()
                    self.manip_time, self.manip_action = get_action_traj(
                        0.0,
                        self.default_motor_pos,
                        self.manip_motor_pos,
                        self.manip_duration,
                        self.control_dt,
                    )
            else:
                self.manip_time, self.manip_action = get_action_traj(
                    0.0,
                    self.default_motor_pos,
                    self.manip_motor_pos,
                    self.manip_duration,
                    self.control_dt,
                )

        if obs.time - self.time_start < self.manip_duration:
            action = np.asarray(
                interpolate_action(
                    obs.time - self.time_start, self.manip_time, self.manip_action
                )
            )
            return {}, action

        self.step_curr += 1

        return control_inputs, motor_target
