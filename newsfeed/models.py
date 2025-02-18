# flake8: noqa: E402
import sys
sys.path.append("..")
from typing import List, Union

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification


class NewsModel:
    def __init__(
        self,
        loss: Union[str, tf.keras.losses.Loss],
        metrics: List[Union[str, tf.keras.metrics.Metric]],
        optimizer,
        num_labels: int = 6,
        model_name: str = "bert-base-uncased",
    ):
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.num_labels = num_labels
        self.model_name = model_name
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )

    def create_and_compile_model(self, learning_rate: float = 5e-5):
        """
        Create and compile the model.
        """
        if isinstance(self.optimizer, str):
            self.optimizer = tf.keras.optimizers.get(
                {
                    "class_name": self.optimizer,
                    "config": {"learning_rate": learning_rate},
                }
            )
        if isinstance(self.loss, str):
            self.loss = tf.keras.losses.get(
                {"class_name": self.loss, "config": {"from_logits": True}}
            )

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

    def fit_model(self, *args, **kwargs) -> tf.keras.callbacks.History:
        """
        Fit the model on the training data.

        Parameters:
        - *args: Positional arguments to pass to model.fit.
        - **kwargs: Keyword arguments to pass to model.fit.

        Returns:
        - A History object containing training details.
        """
        history = self.model.fit(*args, **kwargs)
        return history

    def save_model(self, save_dir: str = "saved_model"):
        """
        Save the model and its weights.
        """
        self.model.save_pretrained(save_dir)
        print(f"Model and weights saved to {save_dir}")

    @classmethod
    def load_model(
        cls,
        save_dir: str,
        loss,
        metrics: List[Union[str, tf.keras.metrics.Metric]],
        optimizer: Union[str, tf.keras.optimizers.Optimizer],
    ):
        """
        Load a saved model.
        """
        model = TFAutoModelForSequenceClassification.from_pretrained(save_dir)
        model_instance = cls(loss, metrics, optimizer)
        model_instance.model = model
        return model_instance
