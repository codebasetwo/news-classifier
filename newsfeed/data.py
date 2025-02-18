# flake8: noqa: E501
import json
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from config import METADATA_DIR, STEMMER, STOPWORDS


def load_dataframe(
    path: str | Path,
    num_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Load Dataset from path.
    Args:
        path (str): data set loacation.
        num_samples (int, optional): The number of samples to load. Default - None.

    Returns:
        pd.Dataframe: dataset of pandas dataframe.
    """
    df = pd.read_csv(path, keep_default_na=False, nrows=num_samples)
    return df


def split_train_test(
    df: pd.DataFrame,
    strat_cols: str = "category",
    test_size: float = 0.2,
    seed=1234,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets."""

    train, val = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle,
        stratify=df[strat_cols],
    )

    train_ds = train.sample(frac=1, random_state=seed)
    val_ds = val.sample(frac=1, random_state=seed)

    return train_ds, val_ds


def tokenize(
    df: pd.DataFrame, max_length: int = 128
) -> Tuple[Dict, tf.Tensor]:
    tokenizer = BertTokenizer.from_pretrained(
        "google-bert/bert-base-uncased", return_dict=False
    )
    encoded_inputs = tokenizer(
        df["headline"].tolist(),
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return dict(
        input_ids=encoded_inputs["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
    ), tf.convert_to_tensor(df["category"])


def clean_text(
    text: str,
    stemmer=STEMMER,
    stopwords: List[str] = STOPWORDS,
) -> str:
    """Clean raw text string."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Remove affixes
    text = stemmer.stem(text)

    # Spacing and filters
    text = re.sub(
        r"([!" "'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text # noqa
    )  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # Remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # Remove multiple spaces
    text = text.strip()  # Strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # Remove links

    return text


def preprocess(
    df: pd.DataFrame, class_index: Dict, max_length: int = 128
) -> Tuple[Dict, tf.Tensor]:
    """Preprocess the data."""
    df.drop_duplicates(
        subset=["headline", "short_description"],
        inplace=True,
        ignore_index=True,
    )
    df["headline"] = df["headline"] + " " + df["short_description"]
    df.drop(
        columns=["link", "short_description", "authors", "date"], inplace=True
    )
    df.dropna(subset=["category"], inplace=True)
    df["headline"] = df["headline"].apply(clean_text)
    df["category"] = df["category"].map(class_index)
    outputs = tokenize(df, max_length)

    return outputs


def user_reader_func(datasets):
    NUM_CORES = os.cpu_count()
    # shuffle the datasets splits
    datasets = datasets.shuffle(NUM_CORES, seed=42)
    # read datasets in parallel and interleave their elements
    return datasets.interleave(
        lambda x: x, num_parallel_calls=tf.data.AUTOTUNE
    )


class DataPreprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        class_to_index: Dict = {},
        max_length: int = 128,
    ) -> None:

        self.df = df
        self.max_length = max_length
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def __call__(self):
        category = self.df["category"].unique().tolist()
        if not self.class_to_index:
            self.class_to_index = {cat: i for i, cat in enumerate(category)}
            self.index_to_class = {
                i: cat for cat, i in self.class_to_index.items()
            }

            # Save the class_to_index to a file
            self._save_class_to_index()
        return self

    def transform(self, batch_size: int = 32) -> tf.data.Dataset:
        outputs = preprocess(self.df, self.class_to_index, self.max_length)
        X_train, y_train = outputs
        outputs = tf.data.Dataset.from_tensor_slices(outputs)
        outputs = outputs.batch(batch_size)
        outputs = outputs.prefetch(tf.data.AUTOTUNE)
        outputs = outputs.snapshot(
            "../datasets/metadata/", reader_func=user_reader_func
        )
        return outputs, X_train, y_train

    def _save_class_to_index(
        self, filepath: Path | str = Path(METADATA_DIR, "class_to_index.json")
    ) -> None:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):  # pragma: no cover
            os.makedirs(directory)

        # Writing dictionary to a JSON file
        with open(filepath, "w") as fp:
            json.dump(self.class_to_index, fp=fp)
            fp.write("\n")

    def __repr__(self):
        return f"DataPreprocessor(df={self.df}, class_to_index={self.class_to_index})"

    @classmethod
    def load_class_index(cls, df: pd.DataFrame) -> "DataPreprocessor":
        with open("../datasets/metadata/class_to_index.json", "r") as file:
            # Extract the dictionary
            class_to_index = json.load(file)

        return cls(df, class_to_index)
