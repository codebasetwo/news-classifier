#!/bin/bash

set -e

# Clean up results directory on script exit or failure
trap 'rm -rf results' EXIT

mkdir results

# Train
export RESULTS_FILE=results/training_results.json
export TRAIN_SET_LOC="datasets/train.csv"
export PARAMS='{"num_epochs": 3, "max_length": 128, "batch_size": 64, "learning_rate": 5e-5}'
python newsfeed/train.py \
    --dataset-loc "$TRAIN_SET_LOC" \
    --params "$PARAMS" \
    --history_fp $RESULTS_FILE \
    --experiment-name "$EXPERIMENT_NAME" \
    

# Evaluate
export RESULTS_FILE=results/evaluation_results.json
export HOLDOUT_LOC="datasets/holdout.csv"
python newsfeed/evaluate.py \
    --experiment_name "$EXPRIMENT_NAME" \
    --test_file_path "$HOLDOUT_LOC" \
    --results_file_path "$RESULTS_FILE" \
