#!/bin/bash

# Run RL experiments sequentially
# Define the different configurations for each experiment

robots=("toddlerbot_2xc")
envs=("walk")
gin_configs=(
    "RewardScales.action_rate=1.0"
    "RewardScales.action_rate=0.5"
    "RewardScales.action_rate=0.2"
    "RewardScales.action_rate=0.1"
)

# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for gin_config in "${gin_configs[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Config: $gin_config"
            
            # Run the Python script with the current configuration
            python toddlerbot/locomotion/train_mjx.py --robot "$robot" --env "$env" --torch --gin-config "$gin_config"

            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done
