"""
Main terrain generator that composes a global heightfield from modular terrain patches.

Each tile in the 2D `terrain_map` is a string specifying a terrain type,
such as "flat", "bumps", "slope", etc. The generator stitches these patches
into a single global heightmap (as an hfield in MuJoCo), returns an MjSpec model,
and optionally loads a robot and defines contact pairs.

Used during simulation setup and RL environment creation.
"""

import xml.etree.ElementTree as ET

import matplotlib.cm as cm
import mujoco
import numpy as np

from toddlerbot.sim.terrain.terrain_types import (
    generate_boxes_patch,
    generate_bumps_patch,
    generate_rough_patch,
    generate_slope_patch,
    generate_stairs_patch,
)

# === Supported terrain types ===
TERRAIN_TYPES = ["flat", "rough", "bump", "slope", "stairs", "boxes"]

FLAT_TERRAIN_THICKNESS = 0.05


def add_debug_spheres_from_hmap(
    spec, global_hmap, nrow, ncol, total_width, total_length, max_height
):
    """Add debug visualization spheres at heightmap positions.

    Creates colored spheres at every heightmap pixel for debugging terrain generation.
    Should only be used with small terrains due to performance impact.

    Args:
        spec: MuJoCo spec to add spheres to.
        global_hmap: The heightmap array.
        nrow: Number of heightmap rows.
        ncol: Number of heightmap columns.
        total_width: Total terrain width.
        total_length: Total terrain length.
        max_height: Maximum terrain height for color scaling.
    """
    cmap = cm.get_cmap("viridis")

    # Correctly calculate step size based on N-1 intervals
    # Handle the edge case of a single row/column to avoid division by zero
    dx = total_width / (ncol - 1) if ncol > 1 else 0
    dy = total_length / (nrow - 1) if nrow > 1 else 0

    for i in range(nrow):
        for j in range(ncol):
            z = float(global_hmap[i, j])
            norm_z = np.clip(z / max_height, 0.0, 1.0)
            color = cmap(norm_z)

            # Calculate position based on grid nodes, not pixel centers
            x = -total_width / 2 + j * dx
            y = -total_length / 2 + i * dy

            # Handle the 1-pixel-wide case
            if ncol == 1:
                x = 0
            if nrow == 1:
                y = 0

            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],
                pos=[x, y, z],
                rgba=[color[0], color[1], color[2], 0.5],
                # group=4,
                name=f"hmap_pt_{i}_{j}",
            )


def create_terrain_spec(
    tile_width,
    tile_length,
    terrain_map,
    robot_xml_path=None,
    robot_position=(0.0, 0.0, 0.315053),
    timestep=0.004,
    pixels_per_meter=16,
    robot_collision_geom_names=None,
    self_contact_pairs=None,
):
    """
    Creates a MuJoCo MjSpec model with procedurally generated terrain and an optional robot.

    Terrain is built from a 2D grid of tile names (e.g., "flat", "slope", "stairs").
    If all tiles are "flat", a single box geom is used instead of a heightfield for performance.

    If a robot is loaded from XML, its initial pose is set, and contact pairs are added between
    terrain geoms and the robot's collision geoms. If robot_collision_geom_names is not provided,
    it is automatically inferred from the geom names appearing in the MJCF <pair> tags.

    Args:
        tile_width (float): Width of each terrain tile in meters.
        tile_length (float): Length of each terrain tile in meters.
        terrain_map (List[List[str]]): 2D list specifying terrain type per tile.
        robot_xml_path (str): Path to MJCF XML file describing the robot (optional).
        robot_position (Tuple[float, float, float]): Initial robot base position.
        timestep (float): Physics simulation timestep.
        pixels_per_meter (int): Resolution scale for the terrain heightfield.
        robot_collision_geom_names (List[str] or None): Names of robot geoms to register
            for terrain contact. If None, inferred from MJCF <pair> entries and filtered
            to geoms defined under the robot body.
        self_contact_pairs (List[List[str]] or None): A list of custom contact pairs (geom1, geom2)
            to be added explicitly to the MuJoCo scene. If defined, this overrides and removes the original
            <contact> section from the robot MJCF.

    Returns:
        spec (MjSpec): Compiled MuJoCo scene specification.
        terrain_geom_names (List[str]): Names of the geoms used for terrain.
        safe_spawns (List[Tuple[float, float, float]]): List of spawn positions per tile.
        global_hmap (np.ndarray): Full assembled heightmap (even for flat terrain).
    """
    # Check if all tiles are flat
    all_flat = all(t == "flat" for row in terrain_map for t in row)

    # Flip vertically so row 0 appears at the top visually
    terrain_map = terrain_map[::-1]

    rows = len(terrain_map)
    cols = len(terrain_map[0])

    total_width = cols * tile_width
    total_length = rows * tile_length

    nrow = int(total_length * pixels_per_meter)
    ncol = int(total_width * pixels_per_meter)

    print("\n=== TERRAIN MAP ===")
    for row in terrain_map:
        print("  - " + " ".join(f"{cell:2}" for cell in row))

    print("\n=== TERRAIN STATS ===")
    print(f"  - Total width  : {total_width:.2f} m")
    print(f"  - Total length : {total_length:.2f} m")
    print(f"  - Pixels per m : {pixels_per_meter}")
    print(f"  - HField size  : {nrow} rows x {ncol} cols")

    # === Create MjSpec model ===
    if robot_xml_path:
        # Load the raw XML string from the file
        with open(robot_xml_path, "r") as f:
            xml_string = f.read()

        # If adding user configured pairs, parse the XML, remove the <contact> section,
        # and load the spec from the modified string.
        if self_contact_pairs is not None:
            tree = ET.fromstring(xml_string)
            contact_element = tree.find("contact")
            if contact_element is not None:
                tree.remove(contact_element)

            modified_xml_string = ET.tostring(tree, encoding="unicode")
            spec = mujoco.MjSpec.from_string(modified_xml_string)
            meshdir = "/".join(robot_xml_path.split("/")[:-1] + ["assets"])

            # Set the mesh directory again since it's loaded from a modified XML string
            spec.meshdir = meshdir

            # Add the custom inter-robot contact pairs
            print("\n=== INTER-ROBOT COLLISION GEOMS ===")
            for pair in self_contact_pairs:
                if len(pair) == 2:
                    spec.add_pair(geomname1=pair[0], geomname2=pair[1])
                    print(f"{pair[0]} <-> {pair[1]}")
                else:
                    raise ValueError(
                        f"Malformed inter-robot contact pair: {pair}. "
                        "Each pair must contain exactly two geom names."
                    )
        else:
            # If not clearing, load the original XML directly.
            spec = mujoco.MjSpec.from_file(robot_xml_path)
    else:
        spec = mujoco.MjSpec()
    spec.option.timestep = timestep

    # --- Place robot (if loaded) ---
    torso = spec.worldbody.first_body()
    torso.pos = list(robot_position)

    # === Add a material for the terrain ===
    spec.add_material(name="terrain_material", rgba=[0.5, 0.5, 0.5, 1])

    # === Lighting and visual config ===
    spec.worldbody.add_light(
        pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )

    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]

    spec.visual.global_.azimuth = 160
    spec.visual.global_.elevation = -20
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    spec.add_texture(
        name="skybox_gradient",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )

    # === Initialize full heightmap ===
    hfield_hmap = np.zeros((nrow, ncol), dtype=np.float32)
    global_hmap = np.zeros((nrow, ncol), dtype=np.float32)
    max_height = 0.0
    safe_spawns = []

    if all_flat:
        # Add flat ground box
        # spec.worldbody.add_geom(
        #     type=mujoco.mjtGeom.mjGEOM_BOX,
        #     size=[total_width / 2, total_length / 2, FLAT_TERRAIN_THICKNESS],
        #     pos=[0, 0, -FLAT_TERRAIN_THICKNESS],
        #     material="terrain_material",
        #     name="flat_geom",
        # )

        # MjSpec deosn't seem to support built-in textures/materials,
        # spec.add_texture(
        #     name="groundplane",
        #     type=mujoco.mjtTexture.mjTEXTURE_2D,
        #     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        #     rgb1=[0.2, 0.3, 0.4],
        #     rgb2=[0.1, 0.2, 0.3],
        #     mark=mujoco.mjtMark.mjMARK_EDGE,
        #     markrgb=[0.8, 0.8, 0.8],
        #     width=300,
        #     height=300,
        # )
        # spec.add_material(
        #     name="groundplane",
        #     textures=["groundplane"],
        #     texuniform=True,
        #     texrepeat=[5, 5],
        #     reflectance=0.0,
        # )

        # Define the checker texture and material as MJCF
        checker_snippet = """
        <mujoco>
            <asset>
                <texture name="groundplane" type="2d" builtin="checker"
                        width="300" height="300"
                        rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                        mark="edge" markrgb="0.8 0.8 0.8"/>

                <material name="groundplane" texture="groundplane"
                        texuniform="false" texrepeat="5 5" reflectance="0.0"/>
            </asset>
        </mujoco>
        """

        # Parse into a temporary MjSpec
        patch_spec = mujoco.MjSpec.from_string(checker_snippet)

        # Add textures
        for tex in patch_spec.textures:
            spec.add_texture(
                name=tex.name,
                type=tex.type,
                builtin=tex.builtin,
                width=tex.width,
                height=tex.height,
                rgb1=tex.rgb1,
                rgb2=tex.rgb2,
                mark=tex.mark,
                markrgb=tex.markrgb,
            )

        # Add materials
        for mat in patch_spec.materials:
            spec.add_material(
                name=mat.name,
                textures=mat.textures,
                texuniform=mat.texuniform,
                texrepeat=mat.texrepeat,
                reflectance=mat.reflectance,
            )

        floor_name = "floor"
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane",
            name=floor_name,
            condim=3,  # default is 1
        )

        # Safe spawns at centers of each tile
        safe_spawns = [
            (
                -total_width / 2 + (j + 0.5) * tile_width,
                -total_length / 2 + (i + 0.5) * tile_length,
                0.0,
            )
            for i in range(rows)
            for j in range(cols)
        ]

        terrain_geom_names = [floor_name]

        # If robot goes out of bounds, (x, y) is clamped and edge heightmap value is used for z check in mjx_env.py.
        hfield_hmap = np.zeros((nrow, ncol), dtype=np.float32)
    else:
        # === Loop through each tile and fill the corresponding patch ===
        terrain_geom_names = []
        for i, row in enumerate(terrain_map):
            for j, terrain in enumerate(row):
                start_row = int(i * tile_length * pixels_per_meter)
                end_row = start_row + int(tile_length * pixels_per_meter)
                start_col = int(j * tile_width * pixels_per_meter)
                end_col = start_col + int(tile_width * pixels_per_meter)

                # Convert tile center to world XY coordinates
                x_center = -total_width / 2 + (j + 0.5) * tile_width
                y_center = -total_length / 2 + (i + 0.5) * tile_length

                # local_patch = None

                # --- Generate the patch ---
                if terrain == "flat":
                    patch = np.zeros(
                        (end_row - start_row, end_col - start_col), dtype=np.float32
                    )

                elif terrain == "bumps":
                    patch, h = generate_bumps_patch(end_row - start_row)
                    max_height = max(max_height, h)

                elif terrain == "rough":
                    patch, h = generate_rough_patch(end_row - start_row)
                    max_height = max(max_height, h)

                elif terrain == "slope":
                    patch, h = generate_slope_patch(end_row - start_row)
                    max_height = max(max_height, h)

                elif terrain == "stairs":
                    patch, h = generate_stairs_patch(end_row - start_row)
                    max_height = max(max_height, h)

                elif terrain == "boxes":
                    patch, h = generate_boxes_patch(end_row - start_row)
                    max_height = max(max_height, h)

                else:
                    raise ValueError(f"Unknown terrain type: '{terrain}'")

                hfield_hmap[start_row:end_row, start_col:end_col] = patch
                global_hmap[start_row:end_row, start_col:end_col] = patch

                # --- Compute safe spawn point at tile center ---
                tile_center_row = start_row + (end_row - start_row) // 2
                tile_center_col = start_col + (end_col - start_col) // 2
                local_z = global_hmap[tile_center_row, tile_center_col]

                safe_spawns.append((x_center, y_center, local_z))

        # === Avoid 0 height if all patches are flat
        if max_height == 0.0:
            max_height = FLAT_TERRAIN_THICKNESS

        # MuJoCo's hfield expects data in [0, 1] and scales it by size[2].
        hfield_min = hfield_hmap.min()
        hfield_max = hfield_hmap.max()

        elevation_range = hfield_max - hfield_min
        if elevation_range <= 1e-6:
            elevation_range = FLAT_TERRAIN_THICKNESS  # Avoid division by zero

        relative_hmap = hfield_hmap - hfield_min
        normalized_hmap = relative_hmap / elevation_range

        # === Define the terrain heightfield ===
        spec.add_hfield(
            name="hfield_terrain",
            size=[
                total_width / 2,  # X radius
                total_length / 2,  # Y radius
                elevation_range,
                0.1,  # Base Z (thickness)
            ],
            nrow=nrow,
            ncol=ncol,
            userdata=normalized_hmap.flatten(),
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="hfield_terrain",
            pos=[0, 0, hfield_min],
            material="terrain_material",
            name="hfield_terrain_geom",
            condim=3,
        )

        terrain_geom_names.append("hfield_terrain_geom")

    if robot_xml_path:
        if robot_collision_geom_names is None:
            # Extract all geoms used in contact pairs
            robot_collision_geom_names = set()

            for pair in spec.pairs:
                if pair.geomname1:
                    robot_collision_geom_names.add(pair.geomname1)
                if pair.geomname2:
                    robot_collision_geom_names.add(pair.geomname2)

    # Add terrain–robot contact pairs
    print("\n=== TERRAIN COLLISION GEOMS ===")
    for terrain_geom in terrain_geom_names:
        for robot_geom in robot_collision_geom_names:
            spec.add_pair(geomname1=terrain_geom, geomname2=robot_geom)
            print(f"  - {robot_geom}")

    # Set to True to visualize the heightmap as spheres
    debug_visualize_hmap = False
    if debug_visualize_hmap and not all_flat:
        add_debug_spheres_from_hmap(
            spec,
            global_hmap,
            nrow,
            ncol,
            total_width,
            total_length,
            max_height,
        )

    return spec, terrain_geom_names, safe_spawns, global_hmap
