"""Crawling locomotion policy for low-profile movement.

This module implements a crawling policy that enables the robot to move
in a prone position using coordinated limb movements.
"""

import numpy as np

from toddlerbot.policies.mjx_policy import MJXPolicy


class CrawlPolicy(MJXPolicy):
    """Crawling policy for the toddlerbot robot."""

    def get_phase_signal(
        self, time_curr: float, init_idx: int = 0, num_frames: int = 950
    ):
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % num_frames

        # Calculate phase based on total_idx
        phase = (total_idx / num_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return phase_signal
