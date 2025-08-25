# test_imu_noise.py
"""Noisy IMU data testing and filtering."""

import time

# imu_noise_numpy.py  — NumPy + SciPy Rotation (bias + colored noise), no delay.
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------- helpers ----------
def rho_from_fc(fc_hz: float, dt: float) -> float:
    # OU→AR(1): rho = exp(-2π f_c dt)
    return float(np.exp(-2.0 * np.pi * fc_hz * dt))


def drive_std_for_target(rho: float, target_std: float) -> float:
    # Var_ar1 = σ_ε^2 / (1 - ρ^2)  -> σ_ε = target_std * sqrt(1-ρ^2)
    return float(target_std * np.sqrt(max(1e-12, 1.0 - rho**2)))


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w], dtype=q_wxyz.dtype)


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=q_xyzw.dtype)


def canon_w_nonneg(q_wxyz: np.ndarray) -> np.ndarray:
    return q_wxyz if q_wxyz[0] >= 0 else -q_wxyz


# ---------- params & state ----------
@dataclass
class IMUNoiseParams:
    dt: float
    # gyro (rad/s)
    gyro_fc: float = 0.35  # low cutoff → slow wander
    gyro_std: float = 0.25  # steady-state std (rad/s)
    gyro_bias_walk_std: float = 2e-4  # per-step RW std
    gyro_white_std: float = 0.0
    # quaternion small-angle rotvec (rad)
    quat_fc: float = 0.25
    quat_std: float = 0.10
    quat_bias_walk_std: float = 1e-4
    quat_white_std: float = 0.0


@dataclass
class IMUNoiseState:
    gyro_ar1: np.ndarray  # (3,)
    gyro_bias: np.ndarray  # (3,)
    quat_ar1: np.ndarray  # (3,)
    quat_bias: np.ndarray  # (3,)


def init_state(dtype=np.float32) -> IMUNoiseState:
    z = np.zeros(3, dtype=dtype)
    return IMUNoiseState(
        gyro_ar1=z.copy(), gyro_bias=z.copy(), quat_ar1=z.copy(), quat_bias=z.copy()
    )


def bno085_defaults(dt: float) -> IMUNoiseParams:
    return IMUNoiseParams(dt=dt)


# ---------- step (no delay, no shocks) ----------
def step(
    rng: np.random.Generator,
    true_gyro_w: np.ndarray,  # (3,) rad/s
    true_quat_wxyz: np.ndarray,  # (4,) wxyz
    st: IMUNoiseState,
    prm: IMUNoiseParams,
) -> Tuple[np.ndarray, np.ndarray, IMUNoiseState]:
    # Gyro AR(1) + bias RW + optional white
    rho_g = rho_from_fc(prm.gyro_fc, prm.dt)
    eps_g_std = drive_std_for_target(rho_g, prm.gyro_std)
    gyro_ar1 = rho_g * st.gyro_ar1 + eps_g_std * rng.standard_normal(3)
    gyro_bias = st.gyro_bias + prm.gyro_bias_walk_std * rng.standard_normal(3)
    gyro_white = prm.gyro_white_std * rng.standard_normal(3)
    gyro_noisy = true_gyro_w + gyro_ar1 + gyro_bias + gyro_white

    # Orientation: AR(1) rotvec + bias RW + optional white
    rho_q = rho_from_fc(prm.quat_fc, prm.dt)
    eps_q_std = drive_std_for_target(rho_q, prm.quat_std)
    quat_ar1 = rho_q * st.quat_ar1 + eps_q_std * rng.standard_normal(3)
    quat_bias = st.quat_bias + prm.quat_bias_walk_std * rng.standard_normal(3)
    quat_white = prm.quat_white_std * rng.standard_normal(3)
    rotvec_total = quat_ar1 + quat_bias + quat_white

    # Compose: R_noise * R_true
    q_true_xyzw = wxyz_to_xyzw(true_quat_wxyz)
    R_noise = R.from_rotvec(rotvec_total)
    R_true = R.from_quat(q_true_xyzw)
    q_noisy_xyzw = (R_noise * R_true).as_quat()
    quat_noisy = canon_w_nonneg(xyzw_to_wxyz(q_noisy_xyzw))

    new_st = IMUNoiseState(
        gyro_ar1=gyro_ar1, gyro_bias=gyro_bias, quat_ar1=quat_ar1, quat_bias=quat_bias
    )
    return (
        gyro_noisy.astype(true_gyro_w.dtype),
        quat_noisy.astype(true_quat_wxyz.dtype),
        new_st,
    )


def simulate_true_signals(t: float):
    """Simple ground-truth IMU: sinusoidal angular vel, integrate to quat."""
    # Angular velocity (rad/s): slow oscillation
    wx = 0.5 * np.sin(0.2 * t)
    wy = 0.3 * np.cos(0.15 * t)
    wz = 0.2 * np.sin(0.1 * t)
    return np.array([wx, wy, wz], dtype=np.float32)


def integrate_quat(quat_wxyz: np.ndarray, ang_vel: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion with angular velocity using small-angle update."""
    rotvec = ang_vel * dt
    dq = R.from_rotvec(rotvec).as_quat()  # xyzw
    q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    q_new = (R.from_quat(dq) * R.from_quat(q_xyzw)).as_quat()
    return np.array([q_new[3], q_new[0], q_new[1], q_new[2]], dtype=np.float32)


if __name__ == "__main__":
    dt = 0.02  # 50 Hz
    rng = np.random.default_rng(0)
    params = bno085_defaults(dt)
    state = init_state()

    # Initialize truth quat (wxyz identity)
    quat_true = np.array([1, 0, 0, 0], dtype=np.float32)

    # Storage for plotting
    T, gyro_true_list, gyro_noisy_list = [], [], []
    euler_true_list, euler_noisy_list = [], []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    t = 0.0
    while True:
        # Simulate ground truth
        gyro_true = simulate_true_signals(t)
        quat_true = integrate_quat(quat_true, gyro_true, dt)

        # Apply noise model
        gyro_noisy, quat_noisy, state = step(rng, gyro_true, quat_true, state, params)

        # Convert to Euler (xyz convention)
        euler_true = R.from_quat(
            [quat_true[1], quat_true[2], quat_true[3], quat_true[0]]
        ).as_euler("xyz", degrees=True)
        euler_noisy = R.from_quat(
            [quat_noisy[1], quat_noisy[2], quat_noisy[3], quat_noisy[0]]
        ).as_euler("xyz", degrees=True)

        # Append to logs
        T.append(t)
        gyro_true_list.append(gyro_true)
        gyro_noisy_list.append(gyro_noisy)
        euler_true_list.append(euler_true)
        euler_noisy_list.append(euler_noisy)

        # Convert to arrays for plotting
        Gt = np.stack(gyro_true_list)
        Gn = np.stack(gyro_noisy_list)
        Et = np.stack(euler_true_list)
        En = np.stack(euler_noisy_list)

        # Plot Euler
        ax1.clear()
        ax1.plot(T, Et[:, 0], "b-", label="Roll True")
        ax1.plot(T, En[:, 0], "b--", label="Roll Noisy")
        ax1.plot(T, Et[:, 1], "g-", label="Pitch True")
        ax1.plot(T, En[:, 1], "g--", label="Pitch Noisy")
        ax1.plot(T, Et[:, 2], "r-", label="Yaw True")
        ax1.plot(T, En[:, 2], "r--", label="Yaw Noisy")
        ax1.set_ylabel("Euler angles (deg)")
        ax1.legend(loc="upper right")
        ax1.set_title("Euler Angles (XYZ)")

        # Plot angular velocity
        ax2.clear()
        ax2.plot(T, Gt[:, 0], "b-", label="wx True")
        ax2.plot(T, Gn[:, 0], "b--", label="wx Noisy")
        ax2.plot(T, Gt[:, 1], "g-", label="wy True")
        ax2.plot(T, Gn[:, 1], "g--", label="wy Noisy")
        ax2.plot(T, Gt[:, 2], "r-", label="wz True")
        ax2.plot(T, Gn[:, 2], "r--", label="wz Noisy")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angular velocity (rad/s)")
        ax2.legend(loc="upper right")
        ax2.set_title("Angular Velocities")

        plt.pause(0.001)

        t += dt
        time.sleep(dt)
