"""Mathematical utilities for signal processing, interpolation, and filtering.

Provides functions for signal generation, trajectory interpolation, coordinate transforms,
and various mathematical operations used throughout the robot control system.
"""

import math
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from scipy.interpolate import interp1d
from scipy.signal import chirp

from toddlerbot.utils.array_utils import ArrayType, R
from toddlerbot.utils.array_utils import array_lib as np


def get_local_vec(world_vec: ArrayType, world_quat: ArrayType) -> ArrayType:
    """Transforms a world-frame vector to local frame using quaternion rotation.

    Args:
        world_vec: Vector in world coordinates.
        world_quat: Quaternion rotation (w, x, y, z format).

    Returns:
        Vector transformed to local frame.
    """
    world_quat_xyzw = np.concatenate(
        [world_quat[..., 1:], world_quat[..., :1]], axis=-1
    )
    world_rot = R.from_quat(world_quat_xyzw)
    world_rot_inv = world_rot.inv()
    # Rotate the vector
    return world_rot_inv.apply(world_vec)


def get_random_sine_signal_config(
    duration: float,
    control_dt: float,
    mean: float,
    frequency_range: List[float],
    amplitude_range: List[float],
):
    """Generates a random sinusoidal signal configuration based on specified parameters.

    Args:
        duration (float): The total duration of the signal in seconds.
        control_dt (float): The time step for signal generation.
        mean (float): The mean value around which the sinusoidal signal oscillates.
        frequency_range (List[float]): A list containing the minimum and maximum frequency values for the signal.
        amplitude_range (List[float]): A list containing the minimum and maximum amplitude values for the signal.

    Returns:
        Tuple[ArrayType, ArrayType]: A tuple containing the time array and the generated sinusoidal signal array.
    """
    frequency = np.random.uniform(*frequency_range)
    amplitude = np.random.uniform(*amplitude_range)

    sine_signal_config: Dict[str, float] = {
        "frequency": frequency,
        "amplitude": amplitude,
        "duration": duration,
        "control_dt": control_dt,
        "mean": mean,
    }

    return sine_signal_config


def get_sine_signal(sine_signal_config: Dict[str, float]):
    """Generate a sine signal based on the provided configuration.

    Args:
        sine_signal_config (Dict[str, float]): Configuration dictionary containing parameters for the sine signal, such as amplitude, frequency, and phase.

    Returns:
        np.ndarray: Array representing the generated sine signal.
    """
    t = np.linspace(
        0,
        sine_signal_config["duration"],
        int(sine_signal_config["duration"] / sine_signal_config["control_dt"]),
        endpoint=False,
        dtype=np.float32,
    )
    signal = sine_signal_config["mean"] + sine_signal_config["amplitude"] * np.sin(
        2 * np.pi * sine_signal_config["frequency"] * t
    )
    return t, signal.astype(np.float32)


def get_chirp_signal(
    duration: float,
    control_dt: float,
    mean: float,
    initial_frequency: float,
    final_frequency: float,
    amplitude: float,
    decay_rate: float,
    method: str = "linear",  # "linear", "quadratic", "logarithmic", etc.
) -> Tuple[ArrayType, ArrayType]:
    """Generate a chirp signal over a specified duration with varying frequency and amplitude.

    Args:
        duration: Total duration of the chirp signal in seconds.
        control_dt: Time step for the signal generation.
        mean: Mean value of the signal.
        initial_frequency: Starting frequency of the chirp in Hz.
        final_frequency: Ending frequency of the chirp in Hz.
        amplitude: Amplitude of the chirp signal.
        decay_rate: Rate at which the amplitude decays over time.
        method: Method of frequency variation, e.g., "linear", "quadratic", "logarithmic".

    Returns:
        A tuple containing:
        - Time array for the chirp signal.
        - Generated chirp signal array.
    """
    t = np.linspace(
        0, duration, int(duration / control_dt), endpoint=False, dtype=np.float32
    )

    # Generate chirp signal without amplitude modulation
    chirp_signal = chirp(
        t, f0=initial_frequency, f1=final_frequency, t1=duration, method=method, phi=-90
    )

    # Apply an amplitude decay envelope based on time (or frequency)
    amplitude_envelope = amplitude * np.exp(-decay_rate * t)

    # Modulate the chirp signal with the decayed amplitude
    signal = mean + amplitude_envelope * chirp_signal

    return t, signal.astype(np.float32)


def round_floats(obj: Any, precision: int = 6) -> Any:
    """
    Recursively round floats in a list-like structure to a given precision.

    Args:
        obj: The list, tuple, or numpy array to round.
        precision (int): The number of decimal places to round to.

    Returns:
        The rounded list, tuple, or numpy array.
    """
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(round_floats(x, precision) for x in obj)
    elif isinstance(obj, np.ndarray):
        return list(np.round(obj, decimals=precision))
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif is_dataclass(obj):
        return type(obj)(  # type: ignore
            **{
                field.name: round_floats(getattr(obj, field.name), precision)
                for field in obj.__dataclass_fields__.values()
            }
        )

    return obj


def round_to_sig_digits(x: float, digits: int):
    """Round a floating-point number to a specified number of significant digits.

    Args:
        x: The number to be rounded.
        digits: The number of significant digits to round to.

    Returns:
        The number rounded to the specified number of significant digits.
    """
    if x == 0.0:
        return 0.0  # Zero is zero in any significant figure
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)


def exponential_moving_average(
    alpha: ArrayType | float,
    current_value: ArrayType | float,
    previous_filtered_value: Optional[ArrayType | float] = None,
) -> ArrayType | float:
    """Calculate the exponential moving average of a current value.

    This function computes the exponential moving average (EMA) for a given current value using a specified smoothing factor, `alpha`. If a previous filtered value is provided, it is used to compute the EMA; otherwise, the current value is used as the initial EMA.

    Args:
        alpha (ArrayType | float): The smoothing factor, where 0 < alpha <= 1.
        current_value (ArrayType | float): The current data point to be filtered.
        previous_filtered_value (Optional[ArrayType | float]): The previous EMA value. If None, the current value is used as the initial EMA.

    Returns:
        ArrayType | float: The updated exponential moving average.
    """
    if previous_filtered_value is None:
        return current_value
    return alpha * current_value + (1 - alpha) * previous_filtered_value


# Recursive Butterworth filter implementation
def butterworth(
    b: ArrayType,
    a: ArrayType,
    x: ArrayType,
    past_inputs: ArrayType,
    past_outputs: ArrayType,
) -> Tuple[ArrayType, ArrayType, ArrayType]:
    """Apply Butterworth filter to input data using filter coefficients.

    Supports both scalar and multi-dimensional inputs.

    Args:
        b: Filter numerator coefficients (b0, b1, ..., bm).
        a: Filter denominator coefficients (a0, a1, ..., an) with a[0] = 1.
        x: Current input value (scalar or array).
        past_inputs: Past input values with shape (filter_order-1, x.shape).
        past_outputs: Past output values with shape (filter_order-1, x.shape).

    Returns:
        tuple: A tuple containing:
            - y: Filtered output with same shape as x
            - new_past_inputs: Updated past inputs
            - new_past_outputs: Updated past outputs
    """
    # Ensure x is at least 1D for consistent operations
    x = np.atleast_1d(x)

    # Compute the current output y[n] based on the difference equation
    # For multi-dimensional inputs, broadcast coefficients appropriately
    b_expanded = b.reshape(-1, *([1] * x.ndim))
    a_expanded = a.reshape(-1, *([1] * x.ndim))

    y = (
        b_expanded[0] * x
        + np.sum(b_expanded[1:] * past_inputs, axis=0)
        - np.sum(a_expanded[1:] * past_outputs, axis=0)
    )

    # Update the state with the new input/output for the next iteration
    new_past_inputs = np.concatenate([x[None], past_inputs[:-1]], axis=0)
    new_past_outputs = np.concatenate([y[None], past_outputs[:-1]], axis=0)

    return y, new_past_inputs, new_past_outputs


def gaussian_basis_functions(phase: ArrayType, N: int = 50):
    """Resample a trajectory to a specified time interval using interpolation.

    Args:
        trajectory (List[Tuple[float, Dict[str, float]]]): The original trajectory, where each element is a tuple containing a timestamp and a dictionary of joint angles.
        desired_interval (float, optional): The desired time interval between resampled points. Defaults to 0.01.
        interp_type (str, optional): The type of interpolation to use ('linear', 'quadratic', 'cubic'). Defaults to 'linear'.

    Returns:
        List[Tuple[float, Dict[str, float]]]: The resampled trajectory with interpolated joint angles at the specified time intervals.
    """
    centers = np.linspace(0, 1, N)
    # Compute the Gaussian basis functions
    basis = np.exp(-np.square(phase - centers) / (2 * N**2))
    return basis


def interpolate(
    p_start: ArrayType | float,
    p_end: ArrayType | float,
    duration: ArrayType | float,
    t: ArrayType | float,
    interp_type: str = "linear",
) -> ArrayType | float:
    """
    Interpolate position at time t using specified interpolation type.

    Args:
        p_start: Initial position.
        p_end: Desired end position.
        duration: Total duration from start to end.
        t: Current time (within 0 to duration).
        interp_type: Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
        Position at time t.
    """
    if t <= 0:
        return p_start

    if t >= duration:
        return p_end

    if interp_type == "linear":
        return p_start + (p_end - p_start) * (t / duration)
    elif interp_type == "quadratic":
        a = (-p_end + p_start) / duration**2
        b = (2 * p_end - 2 * p_start) / duration
        return a * t**2 + b * t + p_start
    elif interp_type == "cubic":
        a = (2 * p_start - 2 * p_end) / duration**3
        b = (3 * p_end - 3 * p_start) / duration**2
        return a * t**3 + b * t**2 + p_start
    else:
        raise ValueError("Unsupported interpolation type: {}".format(interp_type))


def binary_search(arr: ArrayType, t: ArrayType | float) -> int:
    """Performs a binary search on a sorted array to find the index of a target value.

    Args:
        arr (ArrayType): A sorted array of numbers.
        t (ArrayType | float): The target value to search for.

    Returns:
        int: The index of the target value if found; otherwise, the index of the largest element less than the target.
    """
    # Implement binary search using either NumPy or JAX.
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < t:
            low = mid + 1
        elif arr[mid] > t:
            high = mid - 1
        else:
            return mid
    return low - 1


def interpolate_action(
    t: ArrayType | float,
    time_arr: ArrayType,
    action_arr: ArrayType,
    interp_type: str = "linear",
):
    """Interpolates an action value at a given time using specified interpolation method.

    Args:
        t (ArrayType | float): The time at which to interpolate the action.
        time_arr (ArrayType): An array of time points corresponding to the action values.
        action_arr (ArrayType): An array of action values corresponding to the time points.
        interp_type (str, optional): The type of interpolation to use. Defaults to "linear".

    Returns:
        The interpolated action value at time `t`.
    """
    if t <= time_arr[0]:
        return action_arr[0]
    elif t >= time_arr[-1]:
        return action_arr[-1]

    # Use binary search to find the segment containing current_time
    idx = binary_search(time_arr, t)
    idx = max(0, min(idx, len(time_arr) - 2))  # Ensure idx is within valid range

    p_start = action_arr[idx]
    p_end = action_arr[idx + 1]
    duration = time_arr[idx + 1] - time_arr[idx]
    return interpolate(p_start, p_end, duration, t - time_arr[idx], interp_type)


def get_action_traj(
    time_curr: float,
    action_curr: ArrayType,
    action_next: ArrayType,
    duration: float,
    control_dt: float,
    end_time: float = 0.0,
):
    """Calculates the trajectory of an action over a specified duration, interpolating between current and next actions.

    Args:
        time_curr (float): The current time from which the trajectory starts.
        action_curr (ArrayType): The current action state as a NumPy array.
        action_next (ArrayType): The next action state as a NumPy array.
        duration (float): The total duration over which the action should be interpolated.
        end_time (float, optional): The time at the end of the duration where the action should remain constant. Defaults to 0.0.

    Returns:
        Tuple[ArrayType, ArrayType]: A tuple containing the time steps and the corresponding interpolated positions.
    """
    action_time = np.linspace(
        0,
        duration,
        int(duration / control_dt),
        endpoint=True,
        dtype=np.float32,
    )

    action_traj = np.zeros((len(action_time), action_curr.shape[0]), dtype=np.float32)
    for i, t in enumerate(action_time):
        if t < duration - end_time:
            pos = interpolate(
                action_curr,
                action_next,
                duration - end_time,
                t,
            )
        else:
            pos = action_next

        action_traj[i] = pos

    action_time += time_curr

    return action_time, action_traj


def resample_trajectory(
    time_arr: ArrayType,
    trajectory: ArrayType,
    desired_interval: float = 0.02,
    interp_type: str = "linear",
) -> ArrayType:
    """Resamples a trajectory of joint angles over time to a specified time interval using interpolation.

    Args:
        trajectory (List[Tuple[float, Dict[str, float]]]): A list of tuples where each tuple contains a timestamp and a dictionary of joint angles.
        desired_interval (float, optional): The desired time interval between resampled points. Defaults to 0.01.
        interp_type (str, optional): The type of interpolation to use ('linear', etc.). Defaults to 'linear'.

    Returns:
        List[Tuple[float, Dict[str, float]]]: A resampled list of tuples with timestamps and interpolated joint angles.
    """
    # New uniform time array
    new_time_arr = np.arange(time_arr[0], time_arr[-1], desired_interval)

    # Interpolate
    interpolator = interp1d(
        time_arr,
        trajectory,
        kind=interp_type,
        axis=0,
        fill_value="extrapolate",  # or 'bounds_error=False' if you want to be safe
    )
    new_traj = interpolator(new_time_arr)

    return new_time_arr, new_traj
