# flake8: noqa: E501
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from data import DataPreprocessor, load_dataframe
from predict import get_best_model, get_best_run_id
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from snorkel.slicing import PandasSFApplier, slicing_function
from utils import decode, save_dict_to_json

from config import METADATA_DIR, logger


def aggregate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Computes and aggregates evaluation metrics for classification performance.

    This function calculates precision, recall, F1-score (weighted average), and accuracy
    based on the true labels (`y_true`) and predicted labels (`y_pred`). It returns these
    metrics as a dictionary.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values as returned by a classifier.

    Returns:
        Dict[str, float]: A dictionary containing the following metrics:
            - "precision": Weighted average precision score.
            - "recall": Weighted average recall score.
            - "f1_score": Weighted average F1-score.
            - "accuracy": Accuracy score.

    Example:
        >>> y_true = [0, 1, 2, 0, 1]
        >>> y_pred = [0, 2, 1, 0, 0]
        >>> aggregate_metrics(y_true, y_pred)
        {'precision': 0.3, 'recall': 0.4, 'f1_score': 0.33, 'accuracy': 0.4}
    """
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    accuracy = accuracy_score(y_true, y_pred)
    performance = {
        "precision": precision,
        "recall": recall,
        "f1_score": fscore,
        "accuracy": accuracy,
    }

    return performance


def get_per_class_metrics(
    y_true, y_pred, index_to_class: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    """
    Computes per-class classification metrics and returns them in a structured format.

    This function calculates precision, recall, F1-score, and support for each class
    based on the true labels (`y_true`) and predicted labels (`y_pred`). It uses a
    mapping dictionary (`index_to_class`) to decode integer indices to class labels
    and generates a classification report. The report excludes aggregate metrics like
    accuracy, macro average, and weighted average.

    Args:
        y_true (array-like): Ground truth (correct) target values as integer indices.
        y_pred (array-like): Estimated target values as integer indices.
        index_to_class (Dict[int, str]): A dictionary mapping integer indices to their
                                         corresponding class labels.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each key is a class label and
                                      the value is another dictionary containing the
                                      following metrics for that class:
            - "precision": Precision score.
            - "recall": Recall score.
            - "f1-score": F1-score.
            - "support": Number of occurrences in the dataset.

    Example:
        >>> y_true = [0, 1, 2, 0, 1]
        >>> y_pred = [0, 2, 1, 0, 0]
        >>> index_to_class = {0: 'apple', 1: 'banana', 2: 'orange'}
        >>> get_per_class_metrics(y_true, y_pred, index_to_class)
        {
            'apple': {'precision': 0.67, 'recall': 1.0, 'f1-score': 0.8, 'support': 2},
            'banana': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 2},
            'orange': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1}
        }
    """
    labels = decode(y_true, index_to_class)
    labels = list(np.unique(labels))

    class_report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    # Deleting the keys
    del class_report["accuracy"]
    del class_report["macro avg"]
    del class_report["weighted avg"]

    return class_report


@slicing_function()
def headline_less_ten(x):
    """Projects with short titles and descriptions."""
    return len(x.headline.split()) < 10  # less than 10 words


@slicing_function()
def description_less_ten(x):
    """Projects with short titles and descriptions."""
    return len(x.short_description.split()) < 10  # less than 10 words


def get_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame
) -> Dict:
    """Get performance metrics for slices.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        df (Dataset): Ray dataset with labels.
    Returns:
        Dict: performance metrics for slices.
    """
    slice_metrics = {}
    df["headline"] = df["short_description"] + " " + df["headline"]
    slicing_functions = [headline_less_ten, description_less_ten]
    applier = PandasSFApplier(slicing_functions)
    slices = applier.apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)  # create mask
        if sum(mask):  # Check atleast one sample in the slice
            metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            slice_metrics[slice_name] = {}
            slice_metrics[slice_name]["precision"] = metrics[0]
            slice_metrics[slice_name]["recall"] = metrics[1]
            slice_metrics[slice_name]["f1"] = metrics[2]
            slice_metrics[slice_name]["num_samples"] = len(y_true[mask])

    return slice_metrics


def evaluate(
    experiment_name: str,
    test_file_path: str,
    metric: str = "accuracy",
    mode: str = "DESC",
    batch_size: int = 32,
    num_samples: int = None,
    results_file_path: str = str(METADATA_DIR / "evaluations.json"),
    save: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates a trained model on a test dataset and computes performance metrics.

    This function loads a test dataset, preprocesses it, and evaluates the best model
    from a specified experiment. It computes overall metrics, per-class metrics, and
    slice-based metrics. The results can optionally be saved to a JSON file.

    Args:
        experiment_name (str): Name of the experiment to retrieve the best model from.
        test_file_path (str): Path to the test dataset file.
        metric (str, optional): Metric used to determine the best model. Defaults to "accuracy".
        mode (str, optional): Mode for selecting the best model ("ASC" for ascending, "DESC" for descending).
                              Defaults to "DESC".
        batch_size (int, optional): Batch size for inference. Defaults to 32.
        num_samples (int, optional): Number of samples to load from the test dataset. If None, loads all samples.
                                    Defaults to None.
        results_file_path (str, optional): Path to save the evaluation results. Defaults to a predefined path.
        save (bool, optional): Whether to save the evaluation results to a JSON file. Defaults to False.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the following keys:
            - "overall": Dictionary of aggregate metrics (e.g., accuracy, precision, recall, F1-score).
            - "per_class": Dictionary of per-class metrics (e.g., precision, recall, F1-score, support).
            - "slices": Dictionary of slice-based metrics for specific subsets of the data.
    """
    test_ds = load_dataframe(test_file_path, num_samples=num_samples)
    df = test_ds.copy()
    preprocessor = DataPreprocessor.load_class_index(test_ds)
    test_ds, _, y_true = preprocessor().transform(batch_size=batch_size)
    run_id = get_best_run_id(
        experiment_name=experiment_name, metric=metric, mode=mode
    )
    model = get_best_model(run_id)

    # y_pred
    logits = model.predict(test_ds)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    y_pred = np.argmax(probabilities, axis=-1)

    # Evaluate aggregate metrics
    performance = aggregate_metrics(y_true, y_pred)

    # Evaluate per class metrics
    class_report = get_per_class_metrics(
        y_true, y_pred, preprocessor.index_to_class
    )

    # Evaluate slices of data
    slices = get_slice_metrics(y_true, y_pred, df)
    metrics = {
        "overall": performance,
        "per_class": class_report,
        "slices": slices,
    }

    logger.info(json.dumps(metrics, indent=4))
    if save:
        save_dict_to_json(metrics, path=results_file_path)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the model on the test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        required=True,
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        help="File path to test dataset with labels to evaluate on",
        required=True,
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to evaluate the model",
        default="accuracy",
    )
    parser.add_argument(
        "--mode", type=str, help="Mode to evaluate the model", default="DESC"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for evaluation", default=32
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to use for evaluation",
        default=None,
    )
    parser.add_argument(
        "--save",
        type=bool,
        action="store_true",
        help="Save the evaluation results",
        default=True,
    )
    parser.add_argument(
        "--results_file_path",
        type=str,
        help="Path to save the evaluation results",
        default=Path(METADATA_DIR, "evaluations.json"),
    )

    args = parser.parse_args()

    metrics = evaluate(
        experiment_name=args.experiment_name,
        test_file_path=args.test_file_path,
        metric=args.metric,
        mode=args.mode,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        results_file_path=args.results_file_path,
    )
