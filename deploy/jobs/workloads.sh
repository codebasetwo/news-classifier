#!/bin/bash

# Exit immediately if any command fails
set -e

# Run workloads
echo "Running workloads script..."

# Test data
echo "Running tests for data..."
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings | tee 

# Test code
echo "Running tests for code..."
pytest --experiment-name=$EXPERIMENT_NAME tests/code --verbose --disable-warnings | tee 

# Test model
echo "Running tests for models..."
pytest --experiment-name=$EXPERIMENT_NAME tests/models --verbose --disable-warnings | tee

# Build docs (static)
echo "Building documentation..."
mkdocs build