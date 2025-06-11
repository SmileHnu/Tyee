#!/bin/bash

echo "=== Tyee Quick Start: MIT-BIH Arrhythmia Detection Experiment ==="
echo

# Step 1: Create data directory
echo "Step 1/3: Creating data directory..."
mkdir -p ./data/original
echo "Data directory created successfully!"

# Step 2: Download MIT-BIH dataset
echo "Step 2/3: Checking and downloading MIT-BIH dataset..."
cd ./data/original

if ! command -v wget &> /dev/null; then
    echo "Installing wget..."
    apt-get update && apt-get install -y wget
fi

echo "Downloading dataset..."
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
echo "Dataset downloaded successfully!"


# Step 3: Return to project directory and run experiment
cd ../../
echo "Step 3/3: Starting model training..."
python main.py -c config/mit_bih.yaml

echo "=== Experiment completed! ==="
echo "Results saved in ./experiments/ directory"
echo "Use 'tensorboard --logdir ./experiments/' to view training progress"