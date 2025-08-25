#!/usr/bin/env python3
"""
Script to migrate policy log data to motion reference format.

Transforms data from results/*/log_data.lz4 format to motion/*.lz4 format
compatible with the existing motion reference system.
"""

import argparse
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


def migrate_log_data_to_motion(
    log_data_path: str,
    output_path: str,
    robot_name: str = "toddlerbot_2xc",
    start_time: float = 7,
    duration: float = 10,
    control_freq: float = 50,
) -> None:
    """
    Migrate log data from policy execution to motion reference format.

    Args:
        log_data_path: Path to the log_data.lz4 file
        output_path: Path for the output motion file
        robot_name: Name of the robot configuration
    """
    print(f"Loading log data from: {log_data_path}")
    log_data = joblib.load(log_data_path)

    obs_list = log_data["obs_list"]
    action_list = log_data["action_list"]

    obs_list = obs_list[
        start_time * control_freq : (start_time + duration) * control_freq
    ]
    action_list = action_list[
        start_time * control_freq : (start_time + duration) * control_freq
    ]
    for obs in obs_list:
        obs.time -= start_time

    print(f"Found {len(obs_list)} observations and {len(action_list)} actions")

    # Initialize robot and simulation for mujoco.forward rollout
    robot = Robot(robot_name)
    sim = MuJoCoSim(robot, vis_type="none")

    # Prepare data containers
    n_frames = len(obs_list)
    time_data = np.zeros(n_frames, dtype=np.float64)
    action_data = np.zeros((n_frames, len(action_list[0])), dtype=np.float32)

    # Containers for MuJoCo state data
    qpos_data = []
    body_pos_data = []
    body_quat_data = []
    body_lin_vel_data = []
    body_ang_vel_data = []
    site_pos_data = []
    site_quat_data = []

    print("Processing observations and running MuJoCo forward rollout...")

    for i, (obs, action) in enumerate(zip(obs_list, action_list)):
        # Extract time
        time_data[i] = obs.time
        action_data[i] = action

        # Set motor positions and run forward simulation
        motor_angles = dict(zip(robot.motor_ordering, obs.motor_pos))
        motor_angles["left_wrist_pitch_drive"] *= -1
        motor_angles["right_wrist_pitch_drive"] *= -1
        sim.set_motor_angles(motor_angles)
        sim.forward()

        # Extract MuJoCo state data
        qpos = sim.data.qpos.copy()
        qpos[:3] = obs.pos
        qpos[3:7] = obs.rot.as_quat(scalar_first=True)  # wxyz format
        qpos_data.append(qpos)

        # Extract body data
        body_pos = []
        body_quat = []
        body_lin_vel = []
        body_ang_vel = []

        for body_id in range(sim.model.nbody):
            body_pos.append(sim.data.xpos[body_id].copy())
            body_quat.append(sim.data.xquat[body_id].copy())
            body_lin_vel.append(sim.data.cvel[body_id][3:].copy())
            body_ang_vel.append(sim.data.cvel[body_id][:3].copy())

        body_pos_data.append(np.array(body_pos))
        body_quat_data.append(np.array(body_quat))
        body_lin_vel_data.append(np.array(body_lin_vel))
        body_ang_vel_data.append(np.array(body_ang_vel))

        # Extract site data
        site_pos = []
        site_quat = []

        for site_id in range(sim.model.nsite):
            site_pos.append(sim.data.site_xpos[site_id].copy())
            # Site quaternions from rotation matrices
            rot_mat = sim.data.site_xmat[site_id].reshape(3, 3)
            quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
            # Convert to [w, x, y, z] format used by MuJoCo
            quat_mujoco = np.array([quat[3], quat[0], quat[1], quat[2]])
            site_quat.append(quat_mujoco)

        site_pos_data.append(np.array(site_pos))
        site_quat_data.append(np.array(site_quat))

    # Convert lists to numpy arrays
    qpos_data = np.array(qpos_data, dtype=np.float32)
    body_pos_data = np.array(body_pos_data, dtype=np.float32)
    body_quat_data = np.array(body_quat_data, dtype=np.float32)
    body_lin_vel_data = np.array(body_lin_vel_data, dtype=np.float32)
    body_ang_vel_data = np.array(body_ang_vel_data, dtype=np.float32)
    site_pos_data = np.array(site_pos_data, dtype=np.float32)
    site_quat_data = np.array(site_quat_data, dtype=np.float32)

    # Create keyframes from the action data
    print("Creating keyframes...")
    keyframes = []
    for i in range(n_frames):
        keyframe = {
            "name": f"cartwheel_{i:03d}",
            "motor_pos": action_data[i].copy(),
            "joint_pos": np.array(
                list(
                    robot.motor_to_joint_angles(
                        dict(zip(robot.motor_ordering, action_data[i]))
                    ).values()
                ),
                dtype=np.float32,
            ),
            "qpos": qpos_data[i].copy(),
        }
        keyframes.append(keyframe)

    # Create timed sequence (keyframe_name, time_offset pairs)
    timed_sequence = []
    for i, t in enumerate(time_data):
        timed_sequence.append((f"cartwheel_{i:03d}", float(t)))

    # Assemble the motion data dictionary
    motion_data = {
        "time": time_data,
        "qpos": qpos_data,
        "body_pos": body_pos_data,
        "body_quat": body_quat_data,
        "body_lin_vel": body_lin_vel_data,
        "body_ang_vel": body_ang_vel_data,
        "site_pos": site_pos_data,
        "site_quat": site_quat_data,
        "action": action_data,
        "keyframes": keyframes,
        "timed_sequence": timed_sequence,
        "is_robot_relative_frame": False,  # Global frame
    }

    # Save the motion data
    print(f"Saving motion data to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(motion_data, output_path, compress="lz4")

    # Close simulation
    sim.close()

    print("Migration completed successfully!")
    print(f"Generated motion file with {n_frames} frames")
    print(f"Time range: {time_data[0]:.3f}s to {time_data[-1]:.3f}s")


def main():
    """Main entry point for the migration script.

    Parses command line arguments and executes policy-to-motion data migration.
    """
    parser = argparse.ArgumentParser(
        description="Migrate policy log data to motion reference format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/toddlerbot_2xc_cartwheel_mujoco_20250806_200941/log_data.lz4",
        help="Path to input log_data.lz4 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="motion/cartwheel_2xc.lz4",
        help="Path for output motion file",
    )
    parser.add_argument(
        "--robot", type=str, default="toddlerbot_2xc", help="Robot name"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        migrate_log_data_to_motion(
            log_data_path=args.input, output_path=args.output, robot_name=args.robot
        )
        return 0
    except Exception as e:
        print(f"Error during migration: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
