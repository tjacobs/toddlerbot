"""Walking locomotion environment for ToddlerBot.

This module provides the WalkEnv class for training ToddlerBot in bipedal walking.
The environment includes specialized reward functions for stability, gait patterns,
and command following using ZMP (Zero Moment Point) reference trajectories.
"""

from typing import Any, List, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy
from brax import base
from brax.envs.base import State
from jax.scipy.spatial.transform import Rotation as R

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot


class WalkEnv(MJXEnv, env_name="walk"):
    """Walk environment with ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        ref_motion_type: str = "zmp",
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initializes the walking controller with specified configuration and motion reference type.

        Args:
            name (str): The name of the controller.
            robot (Robot): The robot instance to be controlled.
            cfg (MJXConfig): Configuration settings for the controller.
            ref_motion_type (str, optional): Type of motion reference to use. Defaults to 'zmp'.
            fixed_base (bool, optional): Indicates if the robot has a fixed base. Defaults to False.
            add_domain_rand (bool, optional): Whether to add domain randomization. Defaults to True.
            **kwargs (Any): Additional keyword arguments for the superclass initializer.

        Raises:
            ValueError: If an unknown `ref_motion_type` is provided.
        """
        if ref_motion_type == "zmp":
            motion_ref = WalkZMPReference(
                robot,
                cfg.sim.timestep * cfg.action.n_frames,
                cfg.action.cycle_time,
                fixed_base=fixed_base,
            )
        else:
            raise ValueError(f"Unknown ref_motion_type: {ref_motion_type}")

        self.cycle_time = jnp.array(cfg.action.cycle_time)
        self.torso_roll_range = cfg.rewards.torso_roll_range
        self.torso_pitch_range = cfg.rewards.torso_pitch_range

        self.min_feet_y_dist = cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = cfg.rewards.max_feet_y_dist

        super().__init__(
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )

    def visualize_path_frame(self, renderer, pos, rot, axis_len=0.2, alpha=0.5):
        """Visualize coordinate frame axes in the renderer for debugging."""
        colors = {
            "x": [1, 0, 0, alpha],  # red
            "y": [0, 1, 0, alpha],  # green
            "z": [0, 0, 1, alpha],  # blue
        }
        axes = {
            "x": rot.apply(jnp.array([axis_len, 0, 0])),
            "y": rot.apply(jnp.array([0, axis_len, 0])),
            "z": rot.apply(jnp.array([0, 0, axis_len])),
        }

        for key in ["x", "y", "z"]:
            p1 = pos
            p2 = pos + axes[key]
            i = renderer.scene.ngeom
            geom = renderer.scene.geoms[i]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=numpy.zeros(3),
                pos=numpy.zeros(3),
                mat=numpy.eye(3).flatten(),
                rgba=colors[key],
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 5, p1, p2)
            renderer.scene.ngeom += 1

    def render(
        self,
        states: List[State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
    ):
        """Render environment states with path visualization and force arrows."""
        renderer = mujoco.Renderer(self.sys.mj_model, height=height, width=width)
        camera = camera or -1

        push_id = states[0].info["push_id"]
        push_force = numpy.concatenate([states[0].info["push"], numpy.zeros(1)])
        image_list = []
        for state in states:
            d = mujoco.MjData(self.sys.mj_model)
            d.qpos, d.qvel = state.pipeline_state.q, state.pipeline_state.qd
            mujoco.mj_forward(self.sys.mj_model, d)
            renderer.update_scene(d, camera=camera)
            if numpy.linalg.norm(state.info["push"]) > 0:
                push_id = state.info["push_id"]
                push_force = numpy.concatenate([state.info["push"], numpy.zeros(1)])

            self.visualize_path_frame(
                renderer,
                state.info["state_ref"]["path_pos"],
                state.info["state_ref"]["path_rot"],
            )
            self.visualize_force_arrow(renderer, state, push_id, push_force)
            image_list.append(renderer.render())

        return image_list

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        """Generates a random command array based on the provided random number generator state and optionally the last command.

        Args:
            rng (jax.Array): A JAX random number generator state used for sampling.
            last_command (Optional[jax.Array]): The last command array to be used as a reference for generating the new command. If None, a new pose command is sampled.

        Returns:
            jax.Array: A command array consisting of pose and walk/turn components, sampled based on the given probabilities and constraints.
        """
        # Randomly sample an index from the command list
        rng, rng_1, rng_2, rng_3, rng_4, rng_5, rng_6 = jax.random.split(rng, 7)
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])
            # If you want to sample a random pose command for the upper body, uncomment the line below
            pose_command = pose_command.at[:5].set(0.0)

        def sample_walk_command():
            """Generates a random walk command within specified elliptical bounds.

            This function samples random angles to compute a point on an ellipse, ensuring the point lies within defined command ranges. The resulting command is a 3D vector with the z-component set to zero.

            Returns:
                jnp.ndarray: A 3D vector representing the sampled walk command.
            """
            # Sample random angles uniformly between 0 and 2*pi
            theta = jax.random.uniform(rng_3, (1,), minval=0, maxval=2 * jnp.pi)
            # Parametric equation of ellipse
            x_max = jnp.where(
                jnp.sin(theta) > 0, self.command_range[5][1], -self.command_range[5][0]
            )
            x = jax.random.uniform(
                rng_4, (1,), minval=self.deadzone[0], maxval=x_max
            ) * jnp.sin(theta)
            y_max = jnp.where(
                jnp.cos(theta) > 0, self.command_range[6][1], -self.command_range[6][0]
            )
            y = jax.random.uniform(
                rng_4, (1,), minval=self.deadzone[1], maxval=y_max
            ) * jnp.cos(theta)
            z = jnp.zeros(1)
            return jnp.concatenate([x, y, z])

        def sample_turn_command():
            """Generates a sample turn command vector with randomized z-component.

            Returns:
                jnp.ndarray: A concatenated array representing the 3D vector [x, y, z].
            """
            x = jnp.zeros(1)
            y = jnp.zeros(1)
            z = jnp.where(
                jax.random.uniform(rng_5, (1,)) < 0.5,
                jax.random.uniform(
                    rng_6,
                    (1,),
                    minval=self.deadzone[2],
                    maxval=self.command_range[7][1],
                ),
                -jax.random.uniform(
                    rng_6,
                    (1,),
                    minval=self.deadzone[2],
                    maxval=-self.command_range[7][0],
                ),
            )
            return jnp.concatenate([x, y, z])

        random_number = jax.random.uniform(rng_2, (1,))
        walk_command = jnp.where(
            random_number < self.zero_chance,
            jnp.zeros(3),
            jnp.where(
                random_number < self.zero_chance + self.turn_chance,
                sample_turn_command(),
                sample_walk_command(),
            ),
        )
        command = jnp.concatenate([pose_command, walk_command])

        # jax.debug.print("command: {}", command)
        # return jnp.array([0, 0, 0, 0, 0, 0.1, 0.1, 0])
        return command

    # ZMP-based reward functions
    def _reward_torso_roll(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the roll angle of the torso.

        This function computes a reward that penalizes deviations of the torso's roll angle from a specified range. The reward is calculated using an exponential function that decreases as the roll angle moves away from the desired range.

        Args:
            pipeline_state (base.State): The current state of the simulation, containing the orientation of the torso.
            info (dict[str, Any]): Additional information that might be used for reward calculation (not used in this function).
            action (jax.Array): The action taken by the agent (not used in this function).

        Returns:
            jax.Array: A scalar reward value based on the torso's roll angle.
        """
        torso_rot = R.from_quat(
            jnp.concatenate([pipeline_state.x.rot[0][1:], pipeline_state.x.rot[0][:1]])
        )
        torso_roll = torso_rot.as_euler("xyz")[0]

        roll_min = jnp.clip(torso_roll - self.torso_roll_range[0], max=0.0)
        roll_max = jnp.clip(torso_roll - self.torso_roll_range[1], min=0.0)
        reward = (
            jnp.exp(-jnp.abs(roll_min) * 100) + jnp.exp(-jnp.abs(roll_max) * 100)
        ) / 2
        return reward

    def _reward_torso_pitch(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the pitch of the torso.

        This function computes a reward that penalizes deviations of the torso's pitch from a specified range. The reward is calculated using the exponential of the absolute difference between the current pitch and the boundaries of the desired pitch range.

        Args:
            pipeline_state (base.State): The current state of the simulation, containing the rotation quaternion of the torso.
            info (dict[str, Any]): Additional information that might be used for reward computation (not used in this function).
            action (jax.Array): The action taken by the agent (not used in this function).

        Returns:
            jax.Array: A scalar reward value that decreases as the torso pitch deviates from the specified range.
        """
        torso_rot = R.from_quat(
            jnp.concatenate([pipeline_state.x.rot[0][1:], pipeline_state.x.rot[0][:1]])
        )
        torso_pitch = torso_rot.as_euler("xyz")[1]

        pitch_min = jnp.clip(torso_pitch - self.torso_pitch_range[0], max=0.0)
        pitch_max = jnp.clip(torso_pitch - self.torso_pitch_range[1], min=0.0)
        reward = (
            jnp.exp(-jnp.abs(pitch_min) * 100) + jnp.exp(-jnp.abs(pitch_max) * 100)
        ) / 2
        return reward

    def _reward_feet_air_time(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the air time of feet during a movement cycle.

        This function computes a reward for a movement task by evaluating the air time of feet that have made contact with the ground. The reward is only given if the command observation norm exceeds a specified deadzone threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing relevant state information.
            info (dict[str, Any]): A dictionary containing information about the current step, including 'stance_mask', 'last_stance_mask', 'feet_air_time', and 'command_obs'.
            action (jax.Array): The action taken at the current step.

        Returns:
            jax.Array: The computed reward based on the air time of feet that have made contact.
        """
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
        first_contact = (info["feet_air_time"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_time"] * first_contact)
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command_obs"]) > 1e-6
        return reward

    def _reward_feet_clearance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculate the reward for feet clearance during a movement.

        This function computes a reward based on the clearance of feet from the ground,
        considering only the first contact instances and ignoring cases where the command
        magnitude is below a specified threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing information about the current
                state, including 'stance_mask', 'last_stance_mask', 'feet_air_dist', and
                'command_obs'.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: The computed reward for feet clearance.
        """
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
        first_contact = (info["feet_air_dist"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_dist"] * first_contact)
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command_obs"]) > 1e-6
        return reward

    def _reward_feet_distance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the distance between the feet, penalizing positions where the feet are too close or too far apart on the y-axis.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing positional and rotational data.
            info (dict[str, Any]): Additional information that may be used for reward calculation.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: A reward value that decreases as the feet distance deviates from the desired range.
        """
        # Calculates the reward based on the distance between the feet.
        # Penalize feet get close to each other or too far away on the y axis
        torso_rot = R.from_quat(
            jnp.concatenate([pipeline_state.x.rot[0][1:], pipeline_state.x.rot[0][:1]])
        )
        feet_vec = (
            pipeline_state.x.pos[self.feet_link_ids[0]]
            - pipeline_state.x.pos[self.feet_link_ids[1]]
        )
        feet_vec_rotated = torso_rot.inv().apply(feet_vec)
        feet_dist = jnp.abs(feet_vec_rotated[1])
        d_min = jnp.clip(feet_dist - self.min_feet_y_dist, max=0.0)
        d_max = jnp.clip(feet_dist - self.max_feet_y_dist, min=0.0)
        reward = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2
        return reward

    def _reward_feet_slip(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a penalty based on the velocity of feet in contact with the ground.

        This function computes a reward that penalizes high velocities of feet that are in contact with the ground. The penalty is calculated as the negative sum of the squared velocities of the feet in the horizontal plane, weighted by a stance mask indicating which feet are in contact.

        Args:
            pipeline_state (base.State): The current state of the simulation, containing velocity information.
            info (dict[str, Any]): Additional information, including a 'stance_mask' that indicates which feet are in contact with the ground.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: A scalar penalty value representing the negative sum of squared velocities for feet in contact with the ground.
        """
        feet_speed = pipeline_state.xd.vel[self.feet_link_ids]
        feet_speed_square = jnp.square(feet_speed[:, :2])
        reward = -jnp.sum(feet_speed_square * info["stance_mask"])
        # Penalize large feet velocity for feet that are in contact with the ground.
        return reward

    def _reward_stand_still(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a penalty for motion when the command is near zero.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing joint positions.
            info (dict[str, Any]): Additional information, including command observations.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: A negative reward proportional to the squared difference in joint positions, applied when the command is within a specified deadzone.
        """
        # Penalize motion at zero commands
        qpos_diff = jnp.sum(
            jnp.abs(
                pipeline_state.q[self.q_start_idx + self.leg_joint_indices]
                - self.default_qpos[self.q_start_idx + self.leg_joint_indices]
            )
        )
        reward = -(qpos_diff**2)
        reward *= jnp.linalg.norm(info["command_obs"]) < 1e-6
        return reward

    def _reward_align_ground(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the alignment reward for ground contact based on joint positions.

        Args:
            pipeline_state (base.State): The current state of the pipeline containing joint positions.
            info (dict[str, Any]): Additional information, not used in this function.
            action (jax.Array): The action taken, not used in this function.

        Returns:
            jax.Array: The calculated reward based on the alignment of hip, knee, and ankle joint positions.
        """
        hip_pitch_joint_pos = jnp.abs(
            pipeline_state.q[self.q_start_idx + self.hip_pitch_joint_indices]
        )
        knee_joint_pos = jnp.abs(
            pipeline_state.q[self.q_start_idx + self.knee_joint_indices]
        )
        ank_pitch_joint_pos = jnp.abs(
            pipeline_state.q[self.q_start_idx + self.ank_pitch_joint_indices]
        )
        error = hip_pitch_joint_pos + ank_pitch_joint_pos - knee_joint_pos
        reward = -jnp.mean(error**2)
        return reward
