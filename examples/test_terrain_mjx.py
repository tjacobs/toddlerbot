"""
test_terrain_mjx.py

Example MJX script for generating a terrain scene with MjSpec, compiling it,
running a JAX-accelerated MJX simulation step loop, and rendering a video.

Demonstrates:
- Modular terrain creation via create_terrain_spec
- Converting classic MuJoCo model/data to MJX-compatible JAX format
- Running a compiled jit step loop with MJX
- Offscreen rendering with mujoco.Renderer + mediapy video export
"""

import os
import time

import jax
import mediapy as media
import mujoco
import numpy as np
from mujoco import mjx

from toddlerbot.sim.terrain.generate_terrain import create_terrain_spec

# === Terrain config ===
TILE_WIDTH = 4.0
TILE_LENGTH = 4.0
PIXELS_PER_METER = 16
TIMESTEP = 0.004
FRAMERATE = 30
DURATION = 2.0  # seconds

# === Predefined layout ===
TERRAIN_MAP = [
    # ["boxes", "stairs", "bumps"],
    # ["flat", "slope", "rough"],
    ["flat"]
]


def main():
    """Run MJX terrain simulation with JAX-accelerated physics and rendering."""
    # === Paths ===
    this_dir = os.path.dirname(os.path.abspath(__file__))
    robot_path = os.path.join(
        this_dir,
        "../toddlerbot/descriptions/toddlerbot_2xm/toddlerbot_2xm_mjx.xml",
    )

    # === Generate terrain + robot MjSpec ===
    spec, _, _, _ = create_terrain_spec(
        tile_width=TILE_WIDTH,
        tile_length=TILE_LENGTH,
        terrain_map=TERRAIN_MAP,
        robot_xml_path=robot_path,
        timestep=TIMESTEP,
        pixels_per_meter=PIXELS_PER_METER,
    )

    # Add optional camera
    spec.worldbody.add_camera(
        name="perspective",
        pos=[0.7, -0.7, 0.7],
        xyaxes=[1, 1, 0, -1, 1, 3],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )

    # === Compile to classic MuJoCo then convert to MJX ===
    model = spec.compile()
    data = mujoco.MjData(model)

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    print("(mjx_data.qfrc_constraint):", mjx_data.qfrc_constraint.shape)
    print("MJX max contacts (mjx_data.contact.geoms):", mjx_data.contact.geom.shape)
    print("=== Derived collision slots ===")
    print("mj_model.nconmax:", model.nconmax)
    print("mjx_data.qfrc_constraint.shape[0]:", mjx_data.qfrc_constraint.shape[0])
    print("mj_model.npair:", model.npair)
    print("mjx_data.contact.geom.shape[0]:", mjx_data.contact.geom.shape[0])

    # === Setup renderer ===
    renderer = mujoco.Renderer(model, width=640, height=480)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

    jit_step = jax.jit(mjx.step)

    # === Simulate and render ===
    frames = []
    sim_times = []
    render_times = []
    contact_counts = []

    while mjx_data.time < DURATION:
        # Step
        t0 = time.time()
        mjx_data = jit_step(mjx_model, mjx_data)
        t1 = time.time()
        sim_times.append(t1 - t0)

        contact_counts.append(mjx_data.ncon)

        # Render
        if len(frames) < mjx_data.time * FRAMERATE:
            t2 = time.time()
            mj_data = mjx.get_data(model, mjx_data)
            mujoco.mj_forward(model, mj_data)
            renderer.update_scene(
                mj_data, scene_option=scene_option, camera="perspective"
            )
            pixels = renderer.render()
            frames.append(pixels)
            render_times.append(time.time() - t2)

    # === Report ===
    # === Final summary ===
    print("\n=== PERFORMANCE ===")
    print(
        f"First sim step: {sim_times[0] * 1000:.4f} ms, "
        f"Sim step avg: {np.mean(sim_times[1:]) * 1000:.4f} ms, "
        f"std: {np.std(sim_times[1:]) * 1000:.4f} ms"
    )
    print(
        f"First render: {render_times[0] * 1000:.4f} ms, "
        f"Render avg: {np.mean(render_times[1:]) * 1000:.4f} ms, "
        f"std: {np.std(render_times[1:]) * 1000:.4f} ms"
    )
    print(f"Max contacts seen in sim: {max(contact_counts)}")

    # === Save video ===
    out_path = os.path.join(this_dir, "test_terrain_mjx.mp4")
    media.write_video(out_path, frames, fps=FRAMERATE)
    print(f"Saved video to {out_path}")


if __name__ == "__main__":
    main()
