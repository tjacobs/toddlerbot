"""
This script renders a saved motion file (e.g., `.lz4`) for a specified robot using the MuJoCo simulator.
It plays back a sequence of joint positions (`qpos`) and generates a video recording.

Typical usage:
    python motion/render_motion.py --robot toddlerbot_2xc --motion crawl_2xc

"""

import argparse
import os

import joblib
import numpy as np
from tqdm import tqdm

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


def render_motion(robot_name: str, motion_path: str, out_name: str, xml_path: str = ""):
    """Render motion trajectory to video file.

    Loads motion data and generates video recording using MuJoCo simulation.
    """
    robot = Robot(robot_name)
    sim = MuJoCoSim(
        robot, xml_path=xml_path, vis_type="render", fixed_base="fixed" in robot_name
    )

    motion_data = joblib.load(motion_path)
    qpos_seq = motion_data["qpos"]

    print(f"Rendering {len(qpos_seq)} frames from {motion_path}")
    pbar = tqdm(total=len(qpos_seq))

    for qpos in qpos_seq:
        sim.set_qpos(np.array(qpos, dtype=np.float32))
        sim.forward()
        pbar.update(1)

    out_path = os.path.join("motion", f"{out_name}.mp4")
    sim.save_recording("motion", sim.control_dt, 2, f"{out_name}.mp4")
    sim.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="Name of the robot model (e.g., toddlerbot_2xc). Must match your robot MJCF description.",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="crawl_2xc",
        help="Name or path to the motion file (joblib .lz4 format). If no directory is given, looks in 'motion/'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional name of the output video (without extension). Defaults to the motion file's base name.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Scene to load (e.g., 'scene', 'scene_climb_up_box').",
    )
    args = parser.parse_args()

    # Handle .lz4 and motion path
    motion_file = args.motion if args.motion.endswith(".lz4") else args.motion + ".lz4"
    motion_path = (
        motion_file
        if os.path.dirname(motion_file)
        else os.path.join("motion", motion_file)
    )
    out_name = args.out or os.path.splitext(os.path.basename(motion_path))[0]

    if len(args.scene) > 0:
        xml_path = os.path.join(
            "toddlerbot", "descriptions", args.robot, args.scene + ".xml"
        )
    else:
        xml_path = ""

    render_motion(args.robot, motion_path, out_name, xml_path)
