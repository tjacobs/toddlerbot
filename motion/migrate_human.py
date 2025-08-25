"""Script for migrating human motion data to robot-compatible format.

Converts human motion capture data from .pkl format to robot motion reference format,
including trajectory resampling and coordinate frame transformations.
"""

import argparse
import os
import time
from typing import Tuple

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import resample_trajectory


def resample_quat_trajectory(
    time_arr: np.ndarray,
    quat_arr: np.ndarray,
    desired_interval: float = 0.02,
    scalar_first: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample quaternion trajectory using spherical linear interpolation (SLERP).

    Args:
        time_arr (np.ndarray): Array of timestamps of shape (T,)
        quat_arr (np.ndarray): Array of quaternions of shape (T, 4), in (x, y, z, w) format
        desired_interval (float): Desired time interval between resampled points

    Returns:
        Tuple[np.ndarray, np.ndarray]: New time array, interpolated quaternion array (T', 4)
    """
    assert time_arr.ndim == 1
    assert quat_arr.ndim == 2 and quat_arr.shape[1] == 4
    assert len(time_arr) == len(quat_arr)

    # Create new time array
    new_time_arr = np.arange(time_arr[0], time_arr[-1], desired_interval)

    # Convert to Rotation object
    key_rots = R.from_quat(quat_arr, scalar_first=scalar_first)
    slerp = Slerp(time_arr, key_rots)

    # Interpolate
    new_rots = slerp(new_time_arr)

    # Return as (x, y, z, w)
    return new_time_arr, new_rots.as_quat(scalar_first=scalar_first)


def migrate(task, file_path, vis_type, default_fps=30, speed_control=1.0):
    """Migrate human motion data to robot format with trajectory generation.

    Processes human motion capture data and generates robot-compatible trajectory data.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    try:
        data = joblib.load(file_path)
    except Exception as e:
        print(f"Failed to load joblib file: {e}")
        return

    robot = Robot("toddlerbot_2xm")
    sim = MuJoCoSim(robot, vis_type=vis_type)

    mj_joint_indices = np.array(
        [
            mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in robot.joint_ordering
        ]
    )
    mj_joint_indices -= 1

    right_hip_roll_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll"
    )
    right_hip_roll_qpos_adr = sim.model.jnt_qposadr[right_hip_roll_id]

    qpos_arr = np.concatenate(
        [data["root_trans"], data["root_rot"], data["dof"]], axis=-1
    )
    first_qpos = qpos_arr[0, :7]
    root_offset = first_qpos[:3].copy()
    root_offset[2] = 0  # do not offset z
    rot_offset = R.from_quat(first_qpos[3:7], scalar_first=True).inv()
    # Apply transformation to all qpos
    qpos_transformed = qpos_arr.copy()
    for i in range(qpos_arr.shape[0]):
        # Translate root position
        pos = qpos_arr[i, :3] - root_offset
        pos_rot = rot_offset.apply(pos)

        quat = R.from_quat(qpos_arr[i, 3:7], scalar_first=True)
        quat_rot = (rot_offset * quat).as_quat(scalar_first=True)

        qpos_transformed[i, :3] = pos_rot
        qpos_transformed[i, 3:7] = quat_rot

    traj_times = np.arange(len(qpos_transformed)) / default_fps  # fixed at 30Hz
    desired_interval = sim.control_dt * speed_control
    time_replay, qpos_pos = resample_trajectory(
        traj_times, qpos_transformed[:, :3], desired_interval=desired_interval
    )
    _, qpos_quat = resample_quat_trajectory(
        traj_times, qpos_transformed[:, 3:7], desired_interval=desired_interval
    )
    _, qpos_joint = resample_trajectory(
        traj_times, qpos_transformed[:, 7:], desired_interval=desired_interval
    )
    qpos_final = np.concatenate([qpos_pos, qpos_quat, qpos_joint], axis=-1)

    # default_qpos = sim.model.keyframe("home").qpos
    # n_steps = round(1 / sim.control_dt)  # 1 sec for transition
    # qpos_interp = np.tile(default_qpos, (n_steps, 1))
    # qpos_interp[:, :3] = np.linspace(default_qpos[:3], qpos_final[0, :3], n_steps)
    # qpos_interp[:, 7:] = np.linspace(default_qpos[7:], qpos_final[0, 7:], n_steps)
    # qpos_final = np.concatenate([qpos_interp, qpos_final], axis=0)

    for robot_suffix in ["_2xm", "_2xc"]:
        if "2xc" in robot_suffix:
            robot = Robot(f"toddlerbot{robot_suffix}")
            sim = MuJoCoSim(robot, vis_type=vis_type)
            qpos_final[:, right_hip_roll_qpos_adr] *= -1

        qpos_replay = []
        body_pos_replay = []
        body_quat_replay = []
        body_lin_vel_replay = []
        body_ang_vel_replay = []
        site_pos_replay = []
        site_quat_replay = []
        try:
            for qpos in qpos_final:
                sim.set_qpos(qpos)
                # This line makes sure all the motor positions are set correctly
                joint_pos = qpos[7 + mj_joint_indices]
                sim.set_joint_angles(joint_pos)
                sim.forward()

                qpos_replay.append(np.array(sim.data.qpos, dtype=np.float32))
                body_pos_replay.append(np.array(sim.data.xpos, dtype=np.float32))
                body_quat_replay.append(np.array(sim.data.xquat, dtype=np.float32))
                body_lin_vel_replay.append(
                    np.array(sim.data.cvel[:, 3:], dtype=np.float32)
                )
                body_ang_vel_replay.append(
                    np.array(sim.data.cvel[:, :3], dtype=np.float32)
                )
                site_pos = []
                site_quat = []
                for side in ["left", "right"]:
                    for ee_name in ["hand", "foot"]:
                        ee_pos = sim.data.site(f"{side}_{ee_name}_center").xpos.copy()
                        ee_mat = sim.data.site(f"{side}_{ee_name}_center").xmat.reshape(
                            3, 3
                        )
                        ee_quat = R.from_matrix(ee_mat).as_quat(scalar_first=True)
                        site_pos.append(ee_pos)
                        site_quat.append(ee_quat)

                site_pos_replay.append(np.array(site_pos, dtype=np.float32))
                site_quat_replay.append(np.array(site_quat, dtype=np.float32))

                if vis_type == "view":
                    time.sleep(sim.control_dt)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping the replay...")

        finally:
            sim.save_recording(
                "motion", sim.control_dt, 2, f"{args.task}{robot_suffix}.mp4"
            )
            sim.close()

        result_dict = {}
        for key in [
            "time",
            "qpos",
            "body_pos",
            "body_quat",
            "body_lin_vel",
            "body_ang_vel",
            "site_pos",
            "site_quat",
        ]:
            result_dict[key] = np.array(locals()[f"{key}_replay"], dtype=np.float32)

        result_path = os.path.join("motion", f"{task}{robot_suffix}.lz4")
        joblib.dump(result_dict, result_path, compress="lz4")

    print(f"Top-level keys: {list(result_dict.keys())}")
    for k, v in result_dict.items():
        print(f"\nKey: {k}")
        print(f"  Type: {type(v)}")
        if isinstance(v, (np.ndarray, list, tuple)):
            try:
                print(f"  Shape: {np.shape(v)}")
            except Exception:
                print("  (Shape not available)")
        else:
            print("  (No shape info)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect contents of a joblib (.pkl) file."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task name",
        default="cartwheel",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the .pkl or .joblib file",
        default="results/0-Eyes_Japan_Dataset_hamada_turn-04-cartwheels-hamada_poses.pkl",
    )
    parser.add_argument(
        "--vis",
        type=str,
        choices=["none", "view", "render"],
        default="none",
        help="Visualization type: 'none', 'view' for interactive view, or 'render' for rendering to video.",
    )
    args = parser.parse_args()

    migrate(args.task, args.path, args.vis)
