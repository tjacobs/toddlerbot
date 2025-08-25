"""Test IMU sensor readings and real-time visualization.

This module tests the IMU sensor by reading and visualizing real-time orientation and
angular velocity data with optional plotting capabilities.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from toddlerbot.sensing.IMU import ThreadedIMU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Enable real-time plotting")
    args = parser.parse_args()

    if args.plot:
        # Initialize plot
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        euler_lines = (
            ax1.plot([], [], label="Roll", color="r")[0],
            ax1.plot([], [], label="Pitch", color="g")[0],
            ax1.plot([], [], label="Yaw", color="b")[0],
        )
        angvel_raw_lines = (
            ax2.plot([], [], label="X", color="r")[0],
            ax2.plot([], [], label="Y", color="g")[0],
            ax2.plot([], [], label="Z", color="b")[0],
        )
        angvel_filtered_lines = (
            ax3.plot([], [], label="X", color="r")[0],
            ax3.plot([], [], label="Y", color="g")[0],
            ax3.plot([], [], label="Z", color="b")[0],
        )

        ax1.set_ylabel("Euler angles (rad)")
        ax1.legend()
        ax2.set_ylabel("Raw Angular velocity (rad/s)")
        ax2.legend()
        ax2.set_title("Raw Angular Velocity")
        ax3.set_ylabel("Filtered Angular velocity (rad/s)")
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_title("Filtered Angular Velocity (Butterworth)")

    imu = ThreadedIMU()
    imu.start()

    step_times, times, euler_vals, angvel_raw_vals, angvel_filtered_vals = (
        [],
        [],
        [],
        [],
        [],
    )
    start_time = time.monotonic()
    window = 100  # number of points to display
    counter = 0
    try:
        while True:
            step_start = time.monotonic()
            data = imu.get_latest_state()
            step_end = time.monotonic()

            if data is None:
                counter += 1
                print(
                    f"\rWaiting for real-world observation data... [{counter}]",
                    end="",
                    flush=True,
                )
                continue  # Wait for IMU data to be available

            quat, ang_vel_raw, ang_vel_filtered = data
            # print(
            #     f"Data received #{data_count}: quat shape: {quat.shape}, ang_vel_raw shape: {ang_vel_raw.shape}"
            # )
            # print(f"  Quat: {quat}")
            # print(f"  Raw ang_vel: {ang_vel_raw}")
            # print(f"  Filtered ang_vel: {ang_vel_filtered}")
            step_time = step_end - step_start
            step_times.append(step_time)
            # print(f"Step time: {(step_time) * 1000:.3f} ms")
            # print(f"Euler: {rot.as_euler('xyz')}")
            # print(f"Raw Angular velocity: {ang_vel_raw}")
            # print(f"Filtered Angular velocity: {ang_vel_filtered}")

            if args.plot:
                print("Plotting...")
                times.append(step_end - start_time)
                euler_vals.append(R.from_quat(quat, scalar_first=True).as_euler("xyz"))
                angvel_raw_vals.append(ang_vel_raw)
                angvel_filtered_vals.append(ang_vel_filtered)

                # Maintain window size
                if len(times) > window:
                    times = times[-window:]
                    euler_vals = euler_vals[-window:]
                    angvel_raw_vals = angvel_raw_vals[-window:]
                    angvel_filtered_vals = angvel_filtered_vals[-window:]

                euler_np = np.array(euler_vals)
                angvel_raw_np = np.array(angvel_raw_vals)
                angvel_filtered_np = np.array(angvel_filtered_vals)

                for i in range(3):
                    euler_lines[i].set_data(times, euler_np[:, i])
                    angvel_raw_lines[i].set_data(times, angvel_raw_np[:, i])
                    angvel_filtered_lines[i].set_data(times, angvel_filtered_np[:, i])

                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                ax3.relim()
                ax3.autoscale_view()
                plt.pause(0.01)
            else:
                remaining_time = 0.02 - step_time
                print(f"[TEST] Remaining time: {remaining_time:.3f} s")
                time.sleep(max(0, remaining_time))  # Sleep to maintain 50 Hz

    except KeyboardInterrupt:
        pass

    finally:
        # from toddlerbot.utils.misc_utils import dump_profiling_data

        # dump_profiling_data()

        print(
            f"Average step time: {np.mean(step_times) * 1000:.3f} +- {np.std(step_times) * 1000:.3f} ms"
        )

        print("Closing IMU...")
        imu.close()
