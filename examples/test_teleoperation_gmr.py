import argparse
import os
import socket
import time

import numpy as np
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
    draw_frame_batch,
    trans_unity_2_robot,
    yaml_table_2_dict,
)


def is_touch(pos, x_touch_range, y_touch_range):
    return (
        x_touch_range[0] < pos[0] < x_touch_range[1]
        and y_touch_range[0] < pos[1] < y_touch_range[1]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view_frame",
        action="store_true",
        help="Visual frames in mujoco",
    )

    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.75,
        help="The actual height of the human model",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    robot_type = "stanford_toddy_vr"
    src_human = 'human_quest'
    # create a mujoco word loading the toddlerbot model
    if os.name == "nt":
        xml_path = "toddlerbot\\descriptions\\toddlerbot_2xc\\scene_teleop.xml"
        ik_config = "toddlerbot\\manipulation\\teleoperation\\vr_configs\\quest_toddy_gmr.yaml"
        ip_config_path = "toddlerbot\\manipulation\\teleoperation\\vr_configs\\ip_config.yaml"
    elif os.name == "posix":
        xml_path = "toddlerbot/descriptions/toddlerbot_2xc/scene_teleop.xml"
        ik_config = "toddlerbot/manipulation/teleoperation/vr_configs/quest_toddy_gmr.yaml"
        ip_config_path = "toddlerbot/manipulation/teleoperation/vr_configs/ip_config.yaml"
    else:
        raise Exception("Unsupported OS")

    c = pb.connect(pb.DIRECT)
    vis_sp = []
    c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
    for i in range(4):
        vis_sp.append(
            create_primitive_shape(pb, 0.1, pb.GEOM_SPHERE, [0.02], color=c_code[i])
        )
    with open(ip_config_path, "r") as f:
        ip_config = yaml.safe_load(f)
    quest = toddy_quest_module.ToddyQuestBimanualModule(
        ip_config["VR_HOST"],
        ip_config["LOCAL_HOST"],
        ip_config["POSE_CMD_PORT"],
        ip_config["IK_RESULT_PORT"],
        vis_sp=vis_sp,
    )

    retargeter = GMR(
        src_human=src_human,
        tgt_robot=robot_type,
        actual_human_height=args.actual_human_height,
        verbose=False,
    )

    data = retargeter.retarget_data()

    motion_fps = 30

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    # Load the IK config
    with open(ik_config) as f:
        # ik_config = json.load(f)
        ik_config = yaml.safe_load(f)
    ik_match_table = yaml_table_2_dict(ik_config["ik_match_table"])

    anchor_list = ik_config.pop("anchor_list")
    safety_constrainted_link_list = ik_config["safety_constraints"]["safety_list"]

    robot_motion_viewer = RobotMotionViewer(robot_type=robot_type,
                                        motion_fps=motion_fps,
                                        transparent_robot=0,
                                        )

        
    freq = 50.0
    t = 0
    dt = 1.0 / freq
    retargeter.configuration.update_from_keyframe("home")

    initial_t = -1.0
    last_t = initial_t

    # 1. prevent sudden jump of tracked controller position at the begining
    # 2. prevent too aggressive movement of the robot
    x_touch_range = np.array(ik_config["safety_constraints"]["x_touch_range"])
    y_touch_range = np.array(ik_config["safety_constraints"]["y_touch_range"])

    initial_left_ee_pos = np.array(
        ik_config["safety_constraints"]["left_initial_pos"]
    )
    initial_right_ee_pos = np.array(
        ik_config["safety_constraints"]["right_initial_pos"]
    )

    last_left_ee_pos = np.array(ik_config["safety_constraints"]["left_initial_pos"])
    last_right_ee_pos = np.array(
        ik_config["safety_constraints"]["right_initial_pos"]
    )

    delta_pos = np.array(ik_config["safety_constraints"]["delta_pos"])

    initial_pose = {}
    initial_rot = {}
    safety_constrainted_link_pose = {}
    safety_constrainted_link_rot = {}
    # get initial position for anchor joints

    for robot_link, ik_data in ik_match_table.items():
        if robot_link in anchor_list:
            mid = retargeter.model.body(ik_data[0]).mocapid[0]
            initial_pose[robot_link] = (
                retargeter.configuration.get_transform_frame_to_world(
                    robot_link, ik_data[1]
                ).translation()
            )
            wxyz = (
                retargeter.configuration.get_transform_frame_to_world(
                    robot_link, ik_data[1]
                )
                .rotation()
                .wxyz
            )
            data.mocap_pos[mid] = initial_pose[robot_link]
            data.mocap_quat[mid] = wxyz
            xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
            initial_rot[robot_link] = wxyz
        elif robot_link in safety_constrainted_link_list:
            safety_constrainted_link_pose[robot_link] = (
                retargeter.configuration.get_transform_frame_to_world(
                    ik_data[0], "body"
                ).translation()
            )
            wxyz = (
                retargeter.configuration.get_transform_frame_to_world(
                    ik_data[0], "body"
                )
                .rotation()
                .wxyz
            )
            safety_constrainted_link_rot[robot_link] = wxyz
    # get initial head pose, as the head is in world origin of meta quest
    initial_head_pose = retargeter.configuration.get_transform_frame_to_world(
        "head", "body"
    ).translation()
    wxyz = (
        retargeter.configuration.get_transform_frame_to_world("head", "body")
        .rotation()
        .wxyz
    )

    # convert the initial rot to rotation matrix
    initial_head_rot = R.from_quat(wxyz, scalar_first=True).as_matrix()

    gripper_target_mid = retargeter.model.body("left_hand_pose").mocapid[0]

    current_draw_time = time.time()
    last_draw_time = current_draw_time - 0.1
    current_print_time = time.time()
    last_print_time = current_print_time - 0.1
    
    while robot_motion_viewer.is_running():
        step_start = time.time()

        # receive quest robot data
        try:
            raw_string = quest.receive()
            (
                left_hand_pose,
                left_hand_orn,
                right_hand_pose,
                right_hand_orn,
                head_pose,
                head_orn,
            ) = quest.string2pos(raw_string, quest.header)

            head_orn_raw = head_orn

            global_rot = R_y(-90)

            left_hand_pose = global_rot.T @ left_hand_pose
            right_hand_pose = global_rot.T @ right_hand_pose
            head_pose = global_rot.T @ head_pose

            head_pose, head_orn = trans_unity_2_robot(
                head_pose, head_orn, is_quat=True
            )

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
            quest.close()
            break

        current_t = time.time()
        if current_t - initial_t < 1.0:
            # prevent unstable control at the beginning
            last_t = current_t
            continue

        if current_t - last_t > 1.0:
            # restart detect
            last_t = current_t
            initial_t = current_t
            continue

        last_t = current_t
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
        W_pos_lh = ik_config["scales"]["left_hand_center"] * W_pos_lh
        W_pos_rh = ik_config["scales"]["right_hand_center"] * W_pos_rh
        W_pos_head = ik_config["scales"]["head"] * W_pos_head

        W_pos_offset = np.array([-W_pos_head[0], -W_pos_head[1], 0])

        W_pos_lh += W_pos_offset
        W_pos_rh += W_pos_offset
        W_pos_head += W_pos_offset


        if is_touch(W_pos_lh, x_touch_range, y_touch_range):
            W_pos_lh[0] = initial_left_ee_pos[0]
            W_pos_lh[1] = initial_left_ee_pos[1]
        if is_touch(W_pos_rh, x_touch_range, y_touch_range):
            W_pos_rh[0] = initial_right_ee_pos[0]
            W_pos_rh[1] = initial_right_ee_pos[1]

        # convert the rotation matrix to quaternion
        W_rot_lh = R.from_matrix(W_rot_lh).as_quat(scalar_first=True)
        W_rot_rh = R.from_matrix(W_rot_rh).as_quat(scalar_first=True)
        W_rot_head = R.from_matrix(W_rot_head).as_quat(scalar_first=True)

        last_left_ee_pos = W_pos_lh
        last_right_ee_pos = W_pos_rh

        quest_poses = {
            "left_hand_center": [W_pos_lh, W_rot_lh],
            "right_hand_center": [W_pos_rh, W_rot_rh],
            "head": [W_pos_head, W_rot_head],
        }
        # transform the poses from quest frame to mujoco frame
        vicon_data = {}
        # feed data from the quest to vicon data
        for robot_link, ik_data in ik_match_table.items():
            # robot_link of mujoco, ik_data[0] from vicon
            if robot_link in anchor_list:
                vicon_data[ik_data[0]] = [
                    initial_pose[robot_link],
                    initial_rot[robot_link],
                ]
            elif robot_link in quest_poses.keys():
                vicon_data[ik_data[0]] = [
                    quest_poses[robot_link][0],
                    quest_poses[robot_link][1],
                ]
            elif robot_link in safety_constrainted_link_list:
                vicon_data[ik_data[0]] = [
                    safety_constrainted_link_pose[robot_link],
                    safety_constrainted_link_rot[robot_link],
                ]

        # Draw the task targets for reference
        if args.view_frame:
            current_draw_time = time.time()
            poses = []
            rots = []
            sizes = []
            orientaiton_corrections = []
            for robot_link, ik_data in ik_match_table.items():
                if ik_data[0] not in vicon_data:
                    continue
                elif robot_link in ["head", "left_hand_center", "right_hand_center"]:
                    poses.append(
                        ik_config["scale"] * vicon_data[ik_data[0]][0]
                        - retargeter.ground
                    )
                    rots.append(
                        R.from_quat(
                            vicon_data[ik_data[0]][1], scalar_first=True
                        ).as_matrix()
                    )
                    sizes.append(0.1)
                    orientaiton_corrections.append(
                        R.from_quat(ik_data[-1], scalar_first=True)
                    )
            if current_draw_time - last_draw_time > 0.1:
                last_draw_time = current_draw_time
                draw_frame_batch(
                    poses, rots, robot_motion_viewer.viewer, sizes, orientaiton_corrections
                )

        qpos = retargeter.retarget(vicon_data)
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
        )
