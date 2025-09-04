import os
import socket
import time
from typing import Dict, Optional, Tuple

import joblib
import mujoco as mj
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import pybullet as pb
import yaml
from scipy.spatial.transform import Rotation as R

from toddlerbot.manipulation.teleoperation import toddy_quest_module
from toddlerbot.manipulation.teleoperation.general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
)
from toddlerbot.manipulation.teleoperation.general_motion_retargeting import (
    RobotMotionViewer,
)
from toddlerbot.manipulation.teleoperation.rigid_body_sento import (
    create_primitive_shape,
)
from toddlerbot.manipulation.utils.teleop_utils import (
    R_y,
    trans_unity_2_robot,
    yaml_table_2_dict,
)
from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode

# Connect the leader arms to the remote controller and run this script on the remote controller
# during the teleoperation data collection session.


class TeleopVRLeaderPolicy(BasePolicy):
    """Teleoperation leader policy using VR for the leader arms of ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        mujoco_path: str = "toddlerbot/descriptions/toddlerbot_2xc/scene_teleop.xml",
        ik_config: str = "toddlerbot/manipulation/teleoperation/vr_configs/quest_toddy_gmr.yaml",
        ip_config: str = "toddlerbot/manipulation/teleoperation/vr_configs/ip_config.yaml",
        joystick: Optional[Joystick] = None,
        ip: str = "",
        task: str = "",
    ):
        """Initializes the object with specified parameters, setting up the robot's motor positions, joystick, and task-specific configurations.

        Args:
            name (str): The name of the object.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions for the robot.
            joystick (Optional[Joystick]): An optional joystick for manual control. Defaults to None.
            ip (str): The IP address for the ZMQNode. Defaults to an empty string.
            task (str): The task to be performed, affecting motor position setup. Defaults to an empty string.

        Raises:
            ValueError: If the task-specific motion file does not exist.
        """
        super().__init__(name, robot, init_motor_pos)

        self.zmq = ZMQNode(type="sender", ip=ip)

        self.fsr = None
        try:
            self.fsr = FSR()
        except Exception as e:
            print(e)

        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False

        # if joystick is None:
        #     self.joystick = Joystick()
        # else:
        #     self.joystick = joystick

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

        self.left_shoulder_pitch_idx = robot.motor_ordering.index("left_shoulder_pitch")
        self.right_shoulder_pitch_idx = robot.motor_ordering.index("right_shoulder_pitch")

        self.is_prepared = False

        if len(task) > 0:
            self.manip_duration = 2.0

            prep = "kneel" if task == "pick" else "hold"

            motion_file_path = os.path.join("motion", f"{prep}.pkl")
            if os.path.exists(motion_file_path):
                data_dict = joblib.load(motion_file_path)
            else:
                raise ValueError(f"No data files found in {motion_file_path}")

            self.manip_motor_pos = np.array(data_dict["action_traj"], dtype=np.float32)[
                -1
            ][16:30]  # hard coded for toddlerbot_arms

            if task == "hug":
                self.manip_motor_pos[self.left_sho_pitch_idx] -= 0.2
                self.manip_motor_pos[self.right_sho_pitch_idx] += 0.2
        else:
            self.manip_motor_pos = self.default_motor_pos.copy()

        self.ik_config_path = ik_config
        self.ip_config_path = ip_config

        # initialize the mujoco model
        self.model = mj.MjModel.from_xml_path(mujoco_path)
        self.model.opt.gravity[:] = np.array([0, 0, 0])
        self.data = mj.MjData(self.model)

        # initialize the quest robot module
        c = pb.connect(pb.DIRECT)
        vis_sp = []
        c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
        for i in range(4):
            vis_sp.append(
                create_primitive_shape(pb, 0.1, pb.GEOM_SPHERE, [0.02], color=c_code[i])
            )

        with open(self.ip_config_path, "r") as f:
            self.ip_config = yaml.safe_load(f)
        self.quest = toddy_quest_module.ToddyQuestBimanualModule(
            self.ip_config["VR_HOST"],
            self.ip_config["LOCAL_HOST"],
            self.ip_config["POSE_CMD_PORT"],
            self.ip_config["IK_RESULT_PORT"],
            vis_sp=vis_sp,
        )

        self.robot_type = "stanford_toddy_vr"
        self.src_human = "human_quest"

        with open(self.ik_config_path, "r") as f:
            self.ik_config = yaml.safe_load(f)
        self.retargeter = GMR(
            src_human=self.src_human,
            tgt_robot=self.robot_type,
            actual_human_height=self.ik_config["actual_human_height"],
            verbose=False,
        )
        self.rate_limit = self.ik_config.get("rate_limit", False)
        self.data = self.retargeter.retarget_data()

        self.ik_match_table = yaml_table_2_dict(self.ik_config["ik_match_table"])

        self.anchor_list = self.ik_config.pop("anchor_list")
        self.safety_constrainted_link_list = self.ik_config["safety_constraints"][
            "safety_list"
        ]

        # motion_fps = 30
        # self.robot_motion_viewer = RobotMotionViewer(
        #     robot_type=self.robot_type,
        #     motion_fps=motion_fps,
        #     transparent_robot=0,
        # )

        self.init_mujoco_configuration()

        self.last_yaw = None
        self.last_pitch = None
        self.head_control_input = {
            "look_left": 0.0,
            "look_right": 0.0,
            "look_up": 0.0,
            "look_down": 0.0,
        }
        self.yaw_motion_tollerance = 0.025
        self.pitch_motion_tollerance = 0.025

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def is_touch(self, pos, x_touch_range, y_touch_range):
        return (
            x_touch_range[0] < pos[0] < x_touch_range[1]
            and y_touch_range[0] < pos[1] < y_touch_range[1]
        )

    def init_mujoco_configuration(self):
        self.dt = 1.0 / 50.0
        self.retargeter.configuration.update_from_keyframe("home")
        self.initial_pose = {}
        self.initial_rot = {}
        self.safety_constrainted_link_pose = {}
        self.safety_constrainted_link_rot = {}
        self.initial_t = -1.0
        self.last_t = self.initial_t
        # 1. prevent sudden jump of tracked controller position at the begining
        # 2. prevent too aggressive movement of the robot
        self.x_touch_range = np.array(
            self.ik_config["safety_constraints"]["x_touch_range"]
        )
        self.y_touch_range = np.array(
            self.ik_config["safety_constraints"]["y_touch_range"]
        )
        self.initial_left_ee_pos = np.array(
            self.ik_config["safety_constraints"]["left_initial_pos"]
        )
        self.initial_right_ee_pos = np.array(
            self.ik_config["safety_constraints"]["right_initial_pos"]
        )
        self.last_left_ee_pos = np.array(
            self.ik_config["safety_constraints"]["left_initial_pos"]
        )

        self.last_right_ee_pos = np.array(
            self.ik_config["safety_constraints"]["right_initial_pos"]
        )

        self.delta_pos = np.array(self.ik_config["safety_constraints"]["delta_pos"])

        for robot_link, ik_data in self.ik_match_table.items():
            if robot_link in self.anchor_list:
                mid = self.model.body(ik_data[0]).mocapid[0]
                self.initial_pose[robot_link] = (
                    self.retargeter.configuration.get_transform_frame_to_world(
                        robot_link, ik_data[1]
                    ).translation()
                )
                wxyz = (
                    self.retargeter.configuration.get_transform_frame_to_world(
                        robot_link, ik_data[1]
                    )
                    .rotation()
                    .wxyz
                )
                self.data.mocap_pos[mid] = self.initial_pose[robot_link]
                self.data.mocap_quat[mid] = wxyz
                self.initial_rot[robot_link] = wxyz
            elif robot_link in self.safety_constrainted_link_list:
                self.safety_constrainted_link_pose[robot_link] = (
                    self.retargeter.configuration.get_transform_frame_to_world(
                        ik_data[0], "body"
                    ).translation()
                )
                wxyz = (
                    self.retargeter.configuration.get_transform_frame_to_world(
                        ik_data[0], "body"
                    )
                    .rotation()
                    .wxyz
                )
                self.safety_constrainted_link_rot[robot_link] = wxyz

    def get_motor_angles(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current angles of the robot's motors.

        Args:
            type (str): The format in which to return the motor angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The motor angles in the specified format.
            If "dict", returns a dictionary with motor names as keys and angles as values.
            If "array", returns a NumPy array of motor angles.
        """
        joint_angles: Dict[str, float] = {}
        motor_angles: Dict[str, float] = {}
        for name in self.robot.joint_ordering:
            joint_angles[name] = self.data.joint(name).qpos.item()
            motor_angles = self.robot.joint_to_motor_angles(joint_angles)
        if type == "array":
            motor_angle_arr = np.array(list(motor_angles.values()), dtype=np.float32)
            return motor_angle_arr
        else:
            return motor_angles

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Processes the current observation and determines the appropriate action and control inputs.

        Args:
            obs (Obs): The current observation containing time and motor positions.
            is_real (bool, optional): Flag indicating if the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs and the action to be taken.
        """
        try:
            # print("Waiting for data from the quest robot...")
            raw_string = self.quest.receive()
            (
                left_hand_pose,
                left_hand_orn,
                right_hand_pose,
                right_hand_orn,
                head_pose,
                head_orn,
            ) = self.quest.string2pos(raw_string, self.quest.header)

            global_rot = R_y(-90)

            left_hand_pose = global_rot.T @ left_hand_pose
            right_hand_pose = global_rot.T @ right_hand_pose
            head_pose = global_rot.T @ head_pose

            head_pose, head_orn = trans_unity_2_robot(head_pose, head_orn, is_quat=True)

            left_hand_pose, left_hand_orn = trans_unity_2_robot(
                left_hand_pose, left_hand_orn, is_quat=True
            )
            right_hand_pose, right_hand_orn = trans_unity_2_robot(
                right_hand_pose, right_hand_orn, is_quat=True
            )

            head_rot_mapping = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            head_orn = (
                head_rot_mapping
                @ R.from_quat(head_orn).as_matrix()
                @ head_rot_mapping.T
            )
            head_orn = R.from_matrix(head_orn).as_quat()

            lh_rot_mapping = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
            left_hand_orn = (
                lh_rot_mapping
                @ R.from_quat(left_hand_orn).as_matrix()
                @ lh_rot_mapping.T
            )
            left_hand_orn = left_hand_orn @ R_y(-90)
            left_hand_orn = R.from_matrix(left_hand_orn).as_quat()

            rh_rot_mapping = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            right_hand_orn = (
                rh_rot_mapping
                @ R.from_quat(right_hand_orn).as_matrix()
                @ rh_rot_mapping.T
            )
            right_hand_orn = right_hand_orn @ R_y(-90)
            right_hand_orn = R.from_matrix(right_hand_orn).as_quat()

        except socket.error as e:
            print(e)
            pass
        except KeyboardInterrupt:
            self.quest.close()
            return
            # break
        current_t = time.time()

        is_restart = False
        if current_t - self.initial_t < 5.0:
            # prevent unstable control at the beginning
            self.last_t = current_t
            is_restart = True

        if current_t - self.last_t > 1.0:
            # restart detect
            self.last_t = current_t
            self.initial_t = current_t
            is_restart = True

        self.last_t = current_t
        world_origin_pose = np.array(
            [0, 0, 0]
        )  # quest use the character ground as the origin, I guess the height is configured in the hardware
        world_origin_rot = np.eye(3)  # R_z(-90)
        # transform the data in quest frame to mujoco frame
        W_pos_lh = world_origin_rot @ left_hand_pose + world_origin_pose
        W_pos_rh = world_origin_rot @ right_hand_pose + world_origin_pose
        W_pos_head = world_origin_rot @ head_pose + world_origin_pose
        W_rot_lh = world_origin_rot @ R.from_quat(left_hand_orn).as_matrix()
        W_rot_rh = world_origin_rot @ R.from_quat(right_hand_orn).as_matrix()
        W_rot_head = world_origin_rot @ R.from_quat(head_orn).as_matrix()
        # scaling
        W_pos_lh = self.ik_config["scales"]["left_hand_center"] * W_pos_lh
        W_pos_rh = self.ik_config["scales"]["right_hand_center"] * W_pos_rh
        W_pos_head = self.ik_config["scales"]["head"] * W_pos_head

        W_pos_offset = np.array([-W_pos_head[0], -W_pos_head[1], 0])

        W_pos_lh += W_pos_offset
        W_pos_rh += W_pos_offset
        W_pos_head += W_pos_offset

        if self.is_touch(W_pos_lh, self.x_touch_range, self.y_touch_range):
            W_pos_lh[0] = self.initial_left_ee_pos[0]
            W_pos_lh[1] = self.initial_left_ee_pos[1]
        if self.is_touch(W_pos_rh, self.x_touch_range, self.y_touch_range):
            W_pos_rh[0] = self.initial_right_ee_pos[0]
            W_pos_rh[1] = self.initial_right_ee_pos[1]

        # convert the rotation matrix to quaternion
        W_rot_lh = R.from_matrix(W_rot_lh).as_quat(scalar_first=True)
        W_rot_rh = R.from_matrix(W_rot_rh).as_quat(scalar_first=True)
        W_rot_head = R.from_matrix(W_rot_head).as_quat(scalar_first=True)

        if is_restart:
            W_pos_lh = self.initial_left_ee_pos
            W_pos_rh = self.initial_right_ee_pos

        self.last_left_ee_pos = W_pos_lh
        self.last_right_ee_pos = W_pos_rh

        quest_poses = {
            "left_hand_center": [W_pos_lh, W_rot_lh],
            "right_hand_center": [W_pos_rh, W_rot_rh],
            "head": [W_pos_head, W_rot_head],
        }

        vicon_data = {}

        for robot_link, ik_data in self.ik_match_table.items():
            # robot_link of mujoco, ik_data[0] from vicon
            if robot_link in self.anchor_list:
                vicon_data[ik_data[0]] = [
                    self.initial_pose[robot_link],
                    self.initial_rot[robot_link],
                ]
            elif robot_link in quest_poses.keys():
                vicon_data[ik_data[0]] = [
                    quest_poses[robot_link][0],
                    quest_poses[robot_link][1],
                ]
            elif robot_link in self.safety_constrainted_link_list:
                vicon_data[ik_data[0]] = [
                    self.safety_constrainted_link_pose[robot_link],
                    self.safety_constrainted_link_rot[robot_link],
                ]

        # Retarget and pose the model
        qpos = self.retargeter.retarget(vicon_data)
        self.data.qpos[:3] = qpos[:3]
        self.data.qpos[3:7] = qpos[3:7]  # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = qpos[7:]

        mj.mj_forward(self.model, self.data)

        action = self.get_motor_angles("array")


        arm_motor_action = action[self.arm_motor_indices]
        neck_motor_action = action[self.neck_motor_indices]
        
        packed_action = np.concatenate([neck_motor_action, arm_motor_action], axis=0)
        # compile data to send to follower
        msg = ZMQMessage(
            time=time.time(),
            # control_inputs=control_inputs,
            action=packed_action,
            # action=action[self.arm_motor_indices],  # array
            # fsr=np.array([fsrL, fsrR]),
        )
        self.zmq.send_msg(msg)

        return {}, action
