"""Test robot mass calculation in simulation.

This module calculates and displays the total mass of a robot model in MuJoCo simulation.
"""

import argparse

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )

    args = parser.parse_args()

    robot = Robot(args.robot)
    sim = MuJoCoSim(robot, fixed_base="fixed" in args.robot)

    sim.forward()
    mass = sim.model.body_subtreemass[0]

    print(f"Robot: {robot.name}, Mass: {mass:.2f} kg")

    sim.close()
