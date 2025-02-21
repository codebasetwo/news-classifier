#!bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir results

export RESULTS_FILE=results/test_results.txt

# Test data
export DATASET_LOC="../../datasets/holdout.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE


export EXPERIMENT_NAME="news_model"
pytest --experiment-name=$EXPERIMENT_NAME tests/code --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE


# Train
export EXPERIMENT_NAME="news_model_experiment"
export RESULTS_FILE=results/training_results.json
export DATASET_LOC="../../datasets/train.csv"
export PARAMS='{"num_epochs": 3, "max_length": 128, "batch_size": 64, "learning_rate": 5e-5}'
python newsfeed/train.py \
    --dataset-loc "$DATASET_LOC" \
    --params "$PARAMS" \
    --history_fp $RESULTS_FILE \
    --experiment-name "$EXPERIMENT_NAME" \
    

# Evaluate
export RESULTS_FILE=results/evaluation_results.json
export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
python newsfeed/evaluate.py \
    --experiment_name "$EXPRIMENT_NAME" \
    --test_file_path "$HOLDOUT_LOC" \
    --results_file_path "$RESULTS_FILE" \


# Test model
RESULTS_FILE=results/test_model_results.txt
pytest --experiment-name=$EXPERIMENT_NAME tests/models --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE


