# flake8: noqa: E501
import json
from datetime import datetime
from functools import partial
from typing import Dict, Optional

import tensorflow as tf
import utils
from data import DataPreprocessor, load_dataframe, split_train_test
from hyperopt import STATUS_OK, Trials, fmin, tpe
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

import mlflow
from config import SPACE, logger
from models import NewsModel


def train_func(
    params: Dict,
    dataset_loc: str,
    experiment_name: str = "news_model_experiment",
    model_name: str = "bert-base-uncased",
    num_samples: int = None,
    max_len: int = 128,
) -> Dict:
    """
    Trains a model using the specified parameters and dataset.

    This function trains a model (e.g., BERT-based) on the provided dataset. It supports
    optional logging of training history, custom experiment naming, and limiting the
    number of samples for training. The function returns the training results, including
    the final validation loss, training status, and the trained model.

    Args:
        params (Dict): Hyperparameters or configuration for the training process.
        dataset_loc (str): Path to the dataset used for training.
        experiment_name (str, optional): Name of the experiment for tracking purposes.
                                        Defaults to "news_model_experiment".
        model_name (str, optional): Name or identifier of the model architecture to use.
                                   Defaults to "bert-base-uncased".
        num_samples (int, optional): Number of samples to use from the dataset. If None, uses the entire dataset.
                                    Defaults to None.
        max_len (int, optional): Maximum sequence length for input data. Defaults to 128.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - "loss": The final validation loss from the training process.
            - "status": The status of the training process (e.g., STATUS_OK).
            - "model": The trained model.
    """

    params["num_samples"] = num_samples
    params["max_len"] = max_len

    df = load_dataframe(dataset_loc, num_samples=params["num_samples"])
    train_df, val_df = split_train_test(df, test_size=0.3)

    # Get size of training data
    train_size = len(train_df)

    preprocessor = DataPreprocessor(train_df, max_length=params["max_len"])
    train_df, _, _ = preprocessor().transform(batch_size=params["batch_size"])

    preprocessor = DataPreprocessor.load_class_index(val_df)
    val_df, _, _ = preprocessor().transform(batch_size=params["batch_size"])

    # Training components
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    num_train_steps = train_size * params["num_epochs"]

    lr_scheduler = PolynomialDecay(
        initial_learning_rate=params["learning_rate"],
        end_learning_rate=0.0,
        decay_steps=num_train_steps,
    )

    optimizer = Adam(learning_rate=lr_scheduler)
    metrics = [SparseCategoricalAccuracy("accuracy")]

    model = NewsModel(
        loss=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        model_name=model_name,
    )

    model.create_and_compile_model()

    callbacks = [TensorBoard()]

    # Set the experiment name. If it doesn't exist, it will be created.
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        history = model.fit_model(
            train_df,
            validation_dataset=val_df,
            epochs=params["num_epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
        )

        for epoch, (loss, accuracy, val_loss, val_accuracy) in enumerate(
            zip(
                history.history["loss"],
                history.history["accuracy"],
                history.history["val_loss"],
                history.history["val_accuracy"],
            )
        ):
            mlflow.log_metrics(
                {
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

        # Extract a small batch for signature inference
        sample_batch = next(iter(train_df.take(1)))
        input_example = {
            "input_ids": sample_batch[0].numpy(),
            "attention_mask": sample_batch[1].numpy(),
        }  # Input features
        output_example = model.predict(sample_batch)  # Model predictions

        # Infer signature
        signature = infer_signature(input_example, output_example)

        mlflow.tensorflow.log_model(model, "news_model", signature=signature)

    return {
        "loss": history.history["val_loss"][-1],
        "accuracy": history.history["val_accuracy"][-1],
        "status": STATUS_OK,
        "model": model,
    }


def objective(
    params,
    dataset_loc: str,
    experiment_name: str = "news_model_experiment",
    history_fp: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    num_samples: int = None,
    max_len: int = 128,
) -> Dict:

    result = train_func(
        params=params,
        dataset_loc=dataset_loc,
        experiment_name=experiment_name,
        history_fp=history_fp,
        model_name=model_name,
        num_samples=num_samples,
        max_len=max_len,
    )

    return result  # Return the validation loss for Hyperopt


# Hyperopt optimization
def tune_hyperparameters(
    dataset_loc: str,
    experiment_name: str = "news_model_experiment",
    history_fp: str = None,
    model_name: str = "bert-base-uncased",
    num_samples: int = None,
    max_len: int = 128,
):

    trials = Trials()
    wrapped_objective = partial(
        objective,
        dataset_loc=dataset_loc,
        experiment_name=experiment_name,
        history_fp=history_fp,
        model_name=model_name,
        num_samples=num_samples,
        max_len=max_len,
    )

    best = fmin(
        fn=wrapped_objective,
        space=SPACE,  # search space
        algo=tpe.suggest,
        max_evals=10,  # Number of hyperparameter combinations to try
        trials=trials,
    )

    logger.info("Search Done.")

    # Find the details with the minimum loss:
    metadata = min(trials.trials, key=lambda x: x["result"]["loss"])["misc"]

    best_trial = trials.best_trial
    best_trial = best_trial["result"]

    # Save the run data
    data = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "metadata": metadata,
        "params": best,
        "metrics": best_trial,
    }
    logger.info(json.dumps(data, indent=2))

    if history_fp:  # pragma: no cover, saving results
        utils.save_dict_to_json(data, history_fp)

    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="tune.py",
        description="Tune hyperparameters for the news model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_loc", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="news_model_experiment",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--history_fp",
        type=str,
        default=None,
        help="Path to save the training history.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the model.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length of tokenization.",
    )
    args = parser.parse_args()

    best_params = tune_hyperparameters(
        dataset_loc=args.dataset_loc,
        experiment_name=args.experiment_name,
        history_fp=args.history_fp,
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_len=args.max_len,
    )
    print("Best hyperparameters:", best_params)
