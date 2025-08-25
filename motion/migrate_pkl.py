import pickle
import numpy as np
import argparse
import joblib
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="Name of the robot",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="push_up",
        help="Name of the motion file (without .pkl extension)",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)
    sim = MuJoCoSim(robot)
    # Load the original pickle
    data_path = f"motion/{args.motion}.pkl"

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data_len = len(data["keyframes"])
    for i in range(data_len):
        motor_pos = data["keyframes"][i]["motor_pos"]
        motor_angles = dict(zip(robot.motor_ordering, motor_pos))
        for motor_name in robot.motor_ordering:
            # print(motor_name)
            if (
                "shoulder_roll" in motor_name
                or "elbow_roll" in motor_name
                or "hip_pitch" in motor_name
            ):
                motor_angles[motor_name] *= -1

        data["keyframes"][i]["motor_pos"] = np.array(
            list(motor_angles.values()), dtype=np.float32
        )

        sim.set_motor_angles(motor_angles)
        sim.forward()
        qpos = sim.data.qpos.copy()

        data["keyframes"][i]["qpos"] = qpos

        if data["keyframes"][i]["index"] > 0:
            data["keyframes"][i]["name"] += "_" + str(data["keyframes"][i]["index"])

        del data["keyframes"][i]["index"]

    if "sequence" in data:
        data["timed_sequence"] = data["sequence"]
        del data["sequence"]

    joblib.dump(data, f"motion/{args.motion}.lz4", compress="lz4")
