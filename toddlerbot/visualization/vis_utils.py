"""Visualization utilities and helper functions."""

import functools
import inspect
import os
import pickle
import platform
import subprocess
from typing import Any, Callable

import matplotlib.pyplot as plt
import seaborn as sns


def is_x11_available():
    """
    Check if the X11 server is available on the system.

    Returns:
        bool: True if X11 is available, False otherwise.
    """
    # Check if the DISPLAY environment variable is set
    if "DISPLAY" not in os.environ:
        return False

    # Try running a command that requires X11 (e.g., xset)
    try:
        subprocess.run(
            ["xset", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# List of non-interactive backends
non_interactive_backends = ["Agg", "SVG", "PDF", "PS"]

# Check if X11 is available
if platform.system() == "Linux" and not is_x11_available():
    # Switch to a non-interactive backend
    backend_curr = non_interactive_backends[0]  # Use the first non-interactive backend
    plt.switch_backend(backend_curr)
    print(f"X11 is not available. Switched to {backend_curr} backend.")
else:
    print("X11 is available. Using the default backend.")

sns.set_theme(style="darkgrid")


def log_plot_config(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)

        # Extract necessary config values safely
        file_name = params.get("file_name", "plot")
        file_suffix = params.get("file_suffix", "")
        save_path = params.get("save_path", "")
        save_config = params.get("save_config", False)
        blocking = params.get("blocking", True)

        if save_config and save_path:
            if len(file_suffix) > 0:
                file_suffix = f"_{file_suffix}"
            name = f"{file_name}{file_suffix}"

            config = {
                "function": f"{func.__module__}.{func.__name__}",
                "parameters": params,
            }

            os.makedirs(save_path, exist_ok=True)
            config_path = os.path.join(save_path, f"{name}_config.pkl")
            with open(config_path, "wb") as f:
                pickle.dump(config, f)
            print(f"[Visualization] Configuration saved to: {config_path}")

        if blocking:
            return func(*args, **kwargs)
        else:
            return lambda: None

    return wrapper


def make_vis_function(
    func: Callable[..., Any],
    ax: Any = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_path: str = "",
    file_name: str = "",
    file_suffix: str = "",
):
    """Executes a visualization function with specified parameters and optional configuration saving.

    Args:
        func (Callable[..., Any]): The visualization function to execute.
        ax (Any, optional): The axes object for the plot. Defaults to None.
        title (str, optional): The title of the plot. Defaults to an empty string.
        x_label (str, optional): The label for the x-axis. Defaults to an empty string.
        y_label (str, optional): The label for the y-axis. Defaults to an empty string.
        save_config (bool, optional): Whether to save the configuration. Defaults to False.
        save_path (str, optional): The directory path to save the configuration. Defaults to an empty string.
        file_name (str, optional): The name of the file to save the configuration. Defaults to an empty string.
        file_suffix (str, optional): The suffix for the saved file. Defaults to an empty string.
        blocking (bool, optional): Whether the function call should be blocking. Defaults to True.
    """

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs) -> Any | None:
        if len(file_suffix) > 0:
            suffix = f"_{file_suffix}"
        else:
            suffix = ""

        if len(file_name) == 0:
            name = f"{title.lower().replace(' ', '_')}{suffix}"
        else:
            name = f"{file_name}{suffix}"

        # Execute the original function
        result = func(*args, **kwargs)

        if len(title) > 0:
            ax.set_title(title)
        if len(x_label) > 0:
            ax.set_xlabel(x_label)
        if len(y_label) > 0:
            ax.set_ylabel(y_label)

        ax.grid(True)
        plt.tight_layout()

        if save_path is not None:
            if len(save_path) > 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                png_file_path = os.path.join(save_path, f"{name}.png")
                plt.savefig(png_file_path)
                svg_file_path = os.path.join(save_path, f"{name}.svg")
                plt.savefig(svg_file_path)
                print(f"[Visualization] Graph saved as: {png_file_path} and .svg")

            elif plt.get_backend() not in non_interactive_backends:
                plt.show()

        return result

    return wrapped_function


def load_and_run_visualization(config_path: str):
    """Loads a configuration from a pickle file and executes a specified visualization function.

    Args:
        config_path (str): The file path to the pickle configuration file.

    Raises:
        FileNotFoundError: If the configuration file does not exist or is not a pickle file.

    The configuration file must contain a dictionary with the keys:
    - "function": A string specifying the full path of the function to be executed.
    - "parameters": A dictionary of parameters to be passed to the function.
    """
    # Load the configuration based on its type
    if os.path.exists(config_path) and config_path.endswith(".pkl"):
        with open(config_path, "rb") as file:
            config = pickle.load(file)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Dynamically import and call the specified function
    func_module, func_name = config["function"].rsplit(".", 1)
    module = __import__(func_module, fromlist=[func_name])
    func = getattr(module, func_name)
    if "blocking" in config["parameters"]:
        config["parameters"]["blocking"] = True
    if "save_config" in config["parameters"]:
        config["parameters"]["save_config"] = False

    result = func(**config["parameters"])
    if isinstance(result, Callable):
        result()
