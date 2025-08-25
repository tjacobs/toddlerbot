"""
test_terrain.py

Test script for running Toddlerbot simulation on procedurally generated terrain.

This script:
- Defines a manual terrain grid
- Uses create_terrain_spec() to generate a global hfield terrain
- Compiles the model and starts ToddlerbotSimulator
"""

import os

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from toddlerbot.sim.terrain.generate_terrain import create_terrain_spec

# === Terrain config ===
TILE_WIDTH = 4.0
TILE_LENGTH = 4.0
PIXELS_PER_METER = 16
TIMESTEP = 0.004

# === Terrain map ===
TERRAIN_MAP = [
    ["boxes", "stairs", "bumps"],
    ["flat", "slope", "rough"],
]


class ToddlerbotSimulator:
    """Interactive ToddlerBot simulator with keyboard controls."""

    def __init__(self, mj_model, tile_width=4.0, tile_length=4.0):
        self.model = mj_model
        self.data = mujoco.MjData(self.model)

        self.tile_width = tile_width
        self.tile_length = tile_length

        self.MOVE_DISTANCE = 0.05
        self.MOVE_DURATION = 0.05
        self.MOVE_STEPS = int(self.MOVE_DURATION / self.model.opt.timestep)
        self.VELOCITY = self.MOVE_DISTANCE / self.MOVE_DURATION
        self.motion_queue = []

        self.JOINT_MOVE_AMOUNT = 0.4
        self.JOINT_MOVE_DURATION = 0.05
        self.JOINT_MOVE_STEPS = int(self.JOINT_MOVE_DURATION / self.model.opt.timestep)
        self.joint_motion_queue = []

        self.ROTATION_ANGLE = np.deg2rad(10)

        self.KEY_MOVES = {
            glfw.KEY_UP: np.array([1, 0, 0]),
            glfw.KEY_DOWN: np.array([-1, 0, 0]),
            glfw.KEY_LEFT: "turn_left",
            glfw.KEY_RIGHT: "turn_right",
            glfw.KEY_E: np.array([0, 0, 1]),
            glfw.KEY_R: np.array([0, 0, -1]),
        }

        self._init_joint_indices()

    def _init_joint_indices(self):
        """Initialize joint indices for head movement control."""
        yaw_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw_drive"
        )
        pitch_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch_act"
        )
        yaw_qpos_adr = self.model.jnt_qposadr[yaw_id]
        pitch_qpos_adr = self.model.jnt_qposadr[pitch_id]

        self.JOINT_KEYS = {
            glfw.KEY_B: (yaw_qpos_adr, +1),
            glfw.KEY_C: (yaw_qpos_adr, -1),
            glfw.KEY_F: (pitch_qpos_adr, +1),
            glfw.KEY_V: (pitch_qpos_adr, -1),
        }

    def key_callback(self, key):
        """Handle keyboard input for robot movement and head control."""
        if key in self.KEY_MOVES:
            move = self.KEY_MOVES[key]
            if isinstance(move, str):
                quat = self.data.qpos[3:7]
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                delta_rot = R.from_euler(
                    "z",
                    self.ROTATION_ANGLE
                    if move == "turn_left"
                    else -self.ROTATION_ANGLE,
                )
                new_quat = (delta_rot * rot).as_quat()
                self.data.qpos[3:7] = [
                    new_quat[3],
                    new_quat[0],
                    new_quat[1],
                    new_quat[2],
                ]
            else:
                self.motion_queue.append((move, self.MOVE_STEPS))

        elif key in self.JOINT_KEYS:
            joint_idx, direction = self.JOINT_KEYS[key]
            delta = direction * self.JOINT_MOVE_AMOUNT
            self.joint_motion_queue.append((joint_idx, delta, self.JOINT_MOVE_STEPS))

    def _apply_base_motion(self, motion):
        direction, steps_left = motion
        quat = self.data.qpos[3:7]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        world_dir = rot.apply(direction)
        step = world_dir * self.VELOCITY * self.model.opt.timestep
        self.data.qpos[:3] += step
        return (direction, steps_left - 1) if steps_left > 1 else None

    def _apply_joint_motion(self, motion):
        idx, delta_per_step, steps_left = motion
        self.data.qpos[idx] += delta_per_step
        return (idx, delta_per_step, steps_left - 1) if steps_left > 1 else None

    def run(self):
        """Start the interactive simulation with keyboard controls."""
        print("Arrow keys to move, E/R up/down, F/C/V/B for head, ESC to quit")
        current_motion = None
        current_joint_motion = None
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as viewer:
            while viewer.is_running():
                if current_motion is None and self.motion_queue:
                    current_motion = self.motion_queue.pop(0)
                if current_motion:
                    current_motion = self._apply_base_motion(current_motion)

                if current_joint_motion is None and self.joint_motion_queue:
                    joint_idx, total_delta, steps = self.joint_motion_queue.pop(0)
                    delta_per_step = total_delta / steps
                    current_joint_motion = (joint_idx, delta_per_step, steps)
                if current_joint_motion:
                    current_joint_motion = self._apply_joint_motion(
                        current_joint_motion
                    )

                mujoco.mj_forward(self.model, self.data)
                viewer.sync()


def main():
    """Set up terrain and run interactive ToddlerBot simulation."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    robot_path = os.path.join(
        this_dir, "../toddlerbot/descriptions/toddlerbot_2xm/toddlerbot_2xm_mjx.xml"
    )

    # Generate terrain + robot
    spec, _, _, _ = create_terrain_spec(
        tile_width=TILE_WIDTH,
        tile_length=TILE_LENGTH,
        terrain_map=TERRAIN_MAP,
        robot_xml_path=robot_path,
        timestep=TIMESTEP,
        pixels_per_meter=PIXELS_PER_METER,
        # robot_collision_geom_names=[
        #     "left_ankle_roll_link_collision",
        #     "right_ankle_roll_link_collision",
        # ],
        # self_contact_pairs=[["left_hand_collision", "right_hand_collision"]],
    )

    model = spec.compile()
    sim = ToddlerbotSimulator(
        mj_model=model,
        tile_width=TILE_WIDTH,
        tile_length=TILE_LENGTH,
    )
    sim.run()


if __name__ == "__main__":
    main()
