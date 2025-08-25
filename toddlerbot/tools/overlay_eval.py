"""Evaluation tool with overlay visualization."""

import argparse
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np

from toddlerbot.sim.robot import Robot
from toddlerbot.visualization.vis_plot import (
    plot_joint_drive_direction,
    plot_joint_tracking,
)


def load_data(path: str, robot: Robot) -> Dict[str, Dict[str, np.ndarray]]:
    log_data = joblib.load(os.path.join(path, "log_data.lz4"))
    obs_list = log_data["obs_list"]
    action_list = log_data["action_list"]

    data = {}
    data["imu"] = {
        "ang_vel": np.array([obs.ang_vel for obs in obs_list]),
        "euler": np.array([obs.rot.as_euler("xyz", degrees=False) for obs in obs_list]),
    }

    for name in robot.motor_ordering:
        i = robot.motor_ordering.index(name)
        data[name] = {
            "time": np.array([obs.time for obs in obs_list]),
            "pos": np.array([obs.motor_pos[i] for obs in obs_list]),
            "vel": np.array([obs.motor_vel[i] for obs in obs_list]),
            "tor": np.array([obs.motor_tor[i] for obs in obs_list]),
            "action": np.array([act[i] for act in action_list]),
        }
        if obs_list[0].motor_acc is not None:
            data[name]["acc"] = np.array([obs.motor_acc[i] for obs in obs_list])
            data[name]["drive"] = np.sign(data[name]["tor"] * data[name]["acc"])

    return data


def extract_pair_fields(
    data: Dict[str, Dict[str, np.ndarray]], pair: Tuple[str, str]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    field_a, field_b = pair
    time_seq: Dict[str, List[float]] = {}
    data_a: Dict[str, List[float]] = {}
    data_b: Dict[str, List[float]] = {}

    for name in data:
        if name == "imu":
            continue
        time_seq[name] = data[name]["time"].tolist()
        data_a[name] = data[name][field_a].tolist()
        data_b[name] = data[name][field_b].tolist()

    return time_seq, data_a, data_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        required=True,
        help="The name of the run.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["pos:vel", "pos:tor"],
        help="Pairs of fields to plot, e.g., pos:vel pos:tor",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)
    data = load_data(args.path, robot)

    for pair_str in args.fields:
        a, b = pair_str.split(":")

        time_seq, data_a, data_b = extract_pair_fields(data, (a, b))

        if "drive" in b:
            plot_joint_drive_direction(
                time_seq,
                data_a,
                data_b,
                save_path=args.path,
                x_label="Time (s)",
                y_label=f"{a}",
                file_name=f"{a}_vs_{b}",
            )
        else:
            plot_joint_tracking(
                time_seq,
                time_seq,
                data_a,
                data_b,
                save_path=args.path,
                file_name=f"{a}_vs_{b}",
                line_suffix=[f"_{a}", f"_{b}"],
            )


if __name__ == "__main__":
    main()
