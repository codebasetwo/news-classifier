# flake8: noqa: E501
import json
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from data import DataPreprocessor, load_dataframe
from utils import decode

import mlflow
from config import MLFLOW_TRACKING_URI, logger

# Set the tracking uri
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_best_run_id(
    experiment_name: Optional[str] = None,
    metric: str = "accuracy",
    mode: str = "DESC",
) -> str:
    """Retrieves the run ID of the best performing run for a given experiment and metric.

    Searches MLflow runs for the specified experiment, orders them by the given metric
    (either ascending or descending), and returns the run ID of the top-performing run.

    Args:
        experiment_name: The name of the MLflow experiment. If None, searches across all experiments.
        metric: The name of the metric to use for determining the best run (e.g., "accuracy", "loss").
        mode: The sorting mode for the metric ("ASC" for ascending, "DESC" for descending).

    Returns:
        The run ID of the best run.

        Raise:
                ValueError: if `mode` is not "ASC" or "DESC"
                IndexError: If the search results are empty (no runs found).  This is handled specifically because it's a likely scenario.
    """

    if mode not in ("ASC", "DESC"):
        raise ValueError("mode must be either 'ASC' or 'DESC'.")

    if experiment_name is None:
        runs = mlflow.search_runs(order_by=[f"metrics.{metric} {mode}"])
    else:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=[f"metrics.{metric} {mode}"],
        )

    if runs.empty:
        raise IndexError(
            f"No runs found for experiment '{experiment_name or 'all'}' with metric '{metric}'."
        )

    if f"metrics.{metric}" not in runs.columns:
        raise ValueError(f"Metric '{metric}' not found in experiment runs.")

        # Get the best run
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_accuracy = best_run[f"metrics.{metric}"]

    logger.info(f"Best accuracy: {best_accuracy}")

    return best_run_id


def get_best_model(run_id: str) -> tf.keras.Model:
    """Loads the best trained Keras model from a specified MLflow run.

    Args:
        run_id: The ID of the MLflow run containing the desired model.

    Returns:
        The loaded Keras model.
    """
    # Load the best model
    model = mlflow.keras.load_model(f"runs:/{run_id}/model")
    return model


def format_probability(probabilities: np.ndarray, index_to_class: Dict[int, str]) -> Dict[str, float]:
    """
    Formats raw class probabilities into a dictionary mapping class labels to their probabilities.

    This function takes an array of probabilities (e.g., from a softmax output) and a dictionary
    mapping class indices to their corresponding labels. It returns a dictionary where each key
    is a class label, and the value is the probability assigned to that class.

    Args:
        probabilities (np.ndarray): A 1D array of probabilities for each class.
        index_to_class (Dict[int, str]): A dictionary mapping class indices to their corresponding labels.

    Returns:
        Dict[str, float]: A dictionary where keys are class labels and values are their probabilities.

        Example:
        >>> probabilities = np.array([0.1, 0.7, 0.2])
        >>> index_to_class = {0: "cat", 1: "dog", 2: "bird"}
        >>> format_probability(probabilities, index_to_class)
        {"cat": 0.1, "dog": 0.7, "bird": 0.2}

    """
    all_prob = {}
    for i, item in enumerate(probabilities):
        all_prob[index_to_class[i]] = item
    return all_prob


def predict_proba(test_ds: pd.DataFrame, model: tf.keras.Model) -> Dict:
    """
    Generates class probabilities and predictions for a given test dataset using a trained model.

    This function preprocesses the test dataset, computes logits using the provided model,
    and converts the logits into class probabilities using softmax. It then maps the predicted
    class indices to their corresponding labels and returns the results in a structured format.

    Args:
        test_ds (pd.DataFrame): The test dataset to make predictions on.
        model (tf.keras.Model): A trained TensorFlow/Keras model used for inference.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
            - "prediction": The predicted class label.
            - "probabilities": A dictionary mapping class labels to their corresponding probabilities.

    Example:
        >>> test_ds = pd.DataFrame({"text": ["sample text 1", "sample text 2"]})
        >>> model = get_best_model()
        >>> predictions = predict_proba(test_ds, model)
        >>> predictions
        [
            {"prediction": "class_A", "probabilities": {"class_A": 0.85, "class_B": 0.10, "class_C": 0.05}},
            {"prediction": "class_B", "probabilities": {"class_A": 0.10, "class_B": 0.80, "class_C": 0.10}}
        ]
    """

    preprocessor = DataPreprocessor.load_class_index(test_ds)
    class_to_index = preprocessor.class_to_index
    index_to_class = {idx: cls for cls, idx in class_to_index.items()}
    test_ds, _, _ = preprocessor().transform()

    # Prediction(s)
    predictions = model.predict(test_ds)
    # Logits
    logits = predictions.logits

    results = []  # Empty list to store result
    for logit in logits:
        # Convert to probabilities
        probabilities = tf.nn.softmax(logit, axis=-1).numpy()
        # Get the actual class
        indices = np.argmax(probabilities, axis=-1)
        category = decode([indices], index_to_class)[0]
        results.append(
            {
                "prediction": category,
                "probabilities": format_probability(
                    probabilities, index_to_class
                ),
            }
        )

    return results


def predict(
    experiment_name: str,
    metric: str = "accuracy",
    mode: str = "DESC",
    dataset_loc: str = None,
    num_samples: int = None,
) -> List[Dict[str, Union[str, Dict[str, float]]]]:
    """Generates predictions using the best model from a specified MLflow experiment.

    Loads the best performing model (based on the given metric and mode) from the specified MLflow
    experiment, preprocesses the input dataset, generates predictions, and formats the predictions
    into a list of dictionaries.

    Args:
        experiment_name: The name of the MLflow experiment.
        metric: The metric used to determine the best model (e.g., "accuracy", "loss").
        mode: The sorting mode for the metric ("ASC" for ascending, "DESC" for descending).
        dataset_loc: Path to the dataset file for prediction. If None, a sample dataframe is used.
        num_samples: Number of samples to load from the dataset for prediction. If None, all samples are loaded.

    Returns:
        A list of dictionaries. Each dictionary represents a prediction for a single data point
        and contains the following keys:
            - "prediction": The predicted class label (string).
            - "probabilities": A dictionary containing class names as keys and their corresponding
              probabilities (floats) as values.
    """

    if dataset_loc:
        sample = [
            {
                "link": "https://www.huffingtonpost.com/entry/prescription-drug-overdose_us_5b9d7ea7e4b03a1dcc88b84f",
                "headline": 'Texas Court OKs "Up Skirts": WTF?!',
                "category": "NEWS & POLITICS",
                "short_description": "This week, the Texas Court of Appeals made a ruling that is both outrageous and grotesque... and for the first time in recorded history, it has nothing to do with either abortion or the death penalty.",
                "authors": "Jon Hotchkiss, ContributorHost, Be Less Stupid",
                "date": "2014-09-23",
            }
        ]
        sample = pd.read_csv(sample)
    else:
        sample = load_dataframe(dataset_loc, num_samples=num_samples)

    # Get best model run_id
    run_id = get_best_run_id(
        experiment_name=experiment_name, metric=metric, mode=mode
    )
    # Load the best model
    model = get_best_model(run_id=run_id)

    results = predict_proba(sample, model=model)

    logger.info(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="make inference on sample test data,",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="name of experiment to search",
        required=True,
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="metric to use for the search",
        default="accuracy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="how to sort the metric to return",
        default="DESC",
    )
    parser.add_argument(
        "--dataset_loc",
        type=str,
        help="location of user specified data to predict on if passed used the user specified data else uses a defalt sample data.",
        default=None,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="number of samples to read from user specified data if no user is specified ignore",
        default=None,
    )
    args = parser.parse_args()
    results = predict(
        experiment_name=args.experiment_name,
        metric=args.metric,
        mode=args.mode,
        dataset_loc=args.dataset_loc,
        num_samples=args.num_samples,
    )
