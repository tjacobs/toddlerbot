"""Print utility for stereo camera calibration and rectification parameters.

This script loads and displays calibration and rectification parameters from
pickle and npz files for debugging and verification purposes.
"""

import os
import pickle

import numpy as np

# Set print options to suppress scientific notation and set precision
np.set_printoptions(suppress=True, precision=4)

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the calibration files
calibration_file_path = os.path.join(script_dir, "calibration.pkl")
rectification_file_path = os.path.join(script_dir, "rectification.npz")

# --- Load and print from calibration.pkl ---
print("--- Calibration Parameters (from calibration.pkl) ---")
try:
    with open(calibration_file_path, "rb") as f:
        calibration_data = pickle.load(f)
        for key, value in calibration_data.items():
            print(f"{key}:")
            print(value)
            print("-" * 20)
except FileNotFoundError:
    print(f"Error: '{calibration_file_path}' not found.")
except Exception as e:
    print(f"An error occurred while reading the pickle file: {e}")

print("\n" + "=" * 50 + "\n")

# --- Load and print from rectification.npz ---
print("--- Rectification Parameters (from rectification.npz) ---")
try:
    # Load the data (which is a pickled dictionary)
    rectification_data = np.load(rectification_file_path, allow_pickle=True)

    # Iterate directly over the dictionary keys
    for key in rectification_data:
        print(f"{key}:")
        # Access the value using the key
        print(rectification_data[key])
        print("-" * 20)

except FileNotFoundError:
    print(f"Error: '{rectification_file_path}' not found.")
except Exception as e:
    print(f"An error occurred while reading the npz file: {e}")
