"""Cartwheel locomotion environment for ToddlerBot.

This module provides the CartwheelEnv class for training ToddlerBot in cartwheel movements.
The environment extends MJXEnv with cartwheel-specific motion references and command sampling.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.cartwheel_ref import CartwheelReference
from toddlerbot.sim.robot import Robot


class CartwheelEnv(MJXEnv, env_name="cartwheel"):
    """Environment for training cartwheel locomotion with ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initialize the cartwheel environment with motion reference setup."""
        motion_ref = CartwheelReference(
            robot, cfg.sim.timestep * cfg.action.n_frames, fixed_base=fixed_base
        )
        super().__init__(
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        """Sample zero command for cartwheel motion (no external commands needed)."""
        return jnp.zeros(3)
