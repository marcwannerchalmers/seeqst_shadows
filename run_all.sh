#!/bin/bash
set -e

CONFIG_DIR="experiment_cfgs"

for yaml in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$yaml" .yaml)
    echo "Running $name"
    python -u run_experiment.py --config-name="$name"
done