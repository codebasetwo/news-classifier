# flake8: noqa: E501
import json
from datetime import datetime
from typing import Dict, Optional

import tensorflow as tf
import utils
from data import DataPreprocessor, load_dataframe, split_train_test
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

import mlflow
from config import MLFLOW_TRACKING_URI, logger
from models import NewsModel


def train_func(
    dataset_loc: str,
    params: str = None,
    history_fp: Optional[str] = None,
    experiment_name: str = "news_model_experiment",
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    num_samples: int = None,
    batch_size: int = 32,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    split_train: bool = True,
    num_classes: int = 6,
) -> Dict:
    """Trains a text classification model.

    This function trains a model for text classification using the specified
    dataset and hyperparameters.  It supports loading data from a given
    location, uses the Transformers library for model selection, and provides
    options for controlling training parameters like batch size, learning rate,
    and number of epochs.  It also allows for saving training history and
    associating the training run with an experiment name.

    Args:
        dataset_loc: Path to the dataset file. The dataset should be formatted
            appropriately for the chosen model and task (e.g., CSV, JSON, etc.).
        params: Optional path to a JSON file containing additional
            hyperparameters. These parameters will override any individual
            parameters passed directly to the function.
        history_fp: Optional path to save the training history (e.g., loss,
            metrics) to a file (e.g., CSV, JSON). If None, training history is not saved.
        experiment_name: Name of the experiment. This can be used for tracking
            and organizing different training runs.
        model_name: Name of the pre-trained model to use from the Transformers
            library (e.g., "bert-base-uncased", "roberta-base").
        max_length: Maximum sequence length for tokenization.  Longer sequences
            will be truncated.
        num_samples: Optional number of samples to use from the dataset. If None,
            the entire dataset is used. Useful for quick testing or debugging.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        split_train: Whether to split the data into training and validation sets.
            If True, a default split will be used.  Consider providing custom
            splitting logic if needed.
        num_classes: The number of classes in the classification task.

    Returns:
        A dictionary containing the results of the training process. This
        dictionary may include the trained model, final metrics, and other
        relevant information.  The exact contents depend on the specific
        implementation.  It is good practice to include at least the final
        validation loss and any relevant metrics.  Example:
        ```
        {
            "model": <trained model object>,
            "val_loss": 0.5,
            "val_accuracy": 0.85,
            "train_loss": 0.4,
            ...
        }
        ```

    """
    if params:
        params = json.loads(params)
    else:
        params = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "num_samples": num_samples,
        }

    df = load_dataframe(dataset_loc, num_samples=params.get("num_samples", None) or num_samples)
    if split_train:
        train_df, val_df = split_train_test(df)
        # Get size of training data
        train_size = len(train_df)
    else:
        train_df = df
        val_df = None
        train_size = len(train_df)
        validation_split = 0.2

    preprocessor = DataPreprocessor(train_df, max_length=params.get("max_length", None) or max_length)
    train_df, _, _ = preprocessor().transform(batch_size=params.get("batch_size", None) or batch_size)

    if val_df is not None:
        preprocessor = DataPreprocessor.load_class_index(val_df)
        val_df = preprocessor.transform()

    # Training components
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    num_train_steps = train_size * (params.get("num_epochs", None) or num_epochs)

    lr_scheduler = PolynomialDecay(
        initial_learning_rate=params.get("learning_rate", None) or None,
        end_learning_rate=0.0,
        decay_steps=num_train_steps,
    )

    optimizer = Adam(learning_rate=lr_scheduler)
    metrics = [SparseCategoricalAccuracy("accuracy")]

    model = NewsModel(
        loss=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        num_classes=num_classes,
        model_name=params.get("model_name", None) or model_name,
    )

    model.create_and_compile_model()

    callbacks = [TensorBoard()]
    # Set the tracking uri
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Set the experiment name. If it doesn't exist, it will be created.
    mlflow.set_experiment(experiment_name)

    # Start run and log parameters
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        if val_df is not None:
            history = model.fit_model(
                train_df,
                validation_dataset=val_df,
                epochs=params.get("num_epochs", None) or num_epochs,
                batch_size=params.get("batch_size", None) or batch_size,
                callbacks=callbacks,
            )

        else:
            history = model.fit_model(
                train_df,
                validation_split=validation_split,
                epochs=params.get("num_epochs", None) or num_epochs,
                batch_size=params.get("batch_size", None) or batch_size,
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

        # Save the run data
        data = {
            "timestamp": datetime.datetime.now().strftime(
                "%B %d, %Y %I:%M:%S %p"
            ),
            "run_id": run.info.run_id,
            "params": run.data.params,
            "metrics": utils.metrics_by_epoch(history),
        }

        logger.info(json.dumps(data, indent=2))
        if history_fp:  # pragma: no cover, saving results
            utils.save_dict_to_json(data, history_fp)

    logger.info("Training completed.")
    return history.history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a text classification news_model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_loc", type=str, help="Path to the dataset", required=True
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Hyperparameters dictionary for the model if passed ignore passing individual hyperparameters",
        default=None,
    )
    parser.add_argument(
        "--history_fp",
        type=str,
        help="Path to save the training history",
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default="news_model_experiment",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum length of the input token sequence",
        default=128,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to use necceaary for debugging",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training", default=32
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of epochs to train", default=1
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate for the optimizer",
        default=5e-5,
    )
    parser.add_argument(
        "--split_train",
        action="store_false",
        help="Split the dataset into train and validation",
    )
    parser.add_argument(
        "--num_classes", type=int, help="number of classes in data ", default=6
    )

    args = parser.parse_args()
    train_func(
        args.dataset_loc,
        params=args.params,
        history_fp=args.history_fp,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        max_length=args.max_length,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        split_train=args.split_train,
        num_classes=args.num_classes,
    )
