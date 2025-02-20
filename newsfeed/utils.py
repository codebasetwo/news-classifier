# flake8: noqa: E501
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config import DATASET_BASE_PATH


def unzip_file(
    path_to_zip_file: Path, directory_to_extract: Path = DATASET_BASE_PATH
) -> None:
    """Unzip a file.

    Args:
        path_to_zip_file (Path): File to be unzipped.
        directory_to_extract (Path): directory to store extracted file. Default (../datasets/)

    """
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract)


def load_json(path_to_fp: Path) -> Dict:
    """load a json file.

    Args:
        path_to_fp (Path): path to the json file.

    Return:
        Dictionary

    """
    with open(path_to_fp, "rb") as file:
        data = json.load(file)

    return data


# Label to index
def encode(df: pd.DataFrame) -> Dict:
    """
    Encodes string categories in a DataFrame column into index numbers.

    This function takes a DataFrame and extracts unique categories from the specified
    'category' column. It then maps each unique category to a unique integer index,
    creating a dictionary representation of the encoding.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'category' column with
                           string values to be encoded.

    Returns:
        Dict: A dictionary where keys are the unique categories from the 'category'
              column, and values are their corresponding integer indices.

    Example:
        >>> df = pd.DataFrame({'category': ['apple', 'banana', 'apple', 'orange']})
        >>> encode(df)
        {'apple': 0, 'banana': 1, 'orange': 2}
    """

    category = df["category"].unique().tolist()
    class_to_index = {cat: i for i, cat in enumerate(category)}

    return class_to_index

# Index to label
def decode(indices: List[int], index_to_class: Dict[int, str]) -> List[str]:
    """
    Converts a list of integer indices back to their corresponding string labels.

    This function takes a list of integer indices and a mapping dictionary (index-to-label)
    and returns a list of string labels corresponding to the provided indices.

    Args:
        indices (List[int]): A list of integer indices to be converted to string labels.
        index_to_class (Dict[int, str]): A dictionary mapping integer indices to their
                                         corresponding string labels.

    Returns:
        List[str]: A list of string labels corresponding to the input indices.

    Example:
        >>> index_to_class = {0: 'apple', 1: 'banana', 2: 'orange'}
        >>> decode([0, 1, 2], index_to_class)
        ['apple', 'banana', 'orange']
    """
    return [index_to_class[index] for index in indices]


def save_dict_to_json(
    data: Dict, path: str, cls: Any = None, sortkeys: bool = False
) -> None:
    """Save a dictionary to a specific location.

    Args:
        data (Dict): data to save.
        path (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):  # pragma: no cover
        os.makedirs(directory)

    # Check if the file exists
    if not os.path.exists(path):
        # Writing dictionary to a new JSON file
        with open(path, "w") as fp:
            json.dump([data], fp=fp, cls=cls, sort_keys=sortkeys)
            fp.write("\n")
    else:
        # Reading the existing data from the JSON file
        with open(path, "r") as fp:
            json_file = json.load(fp)

        # Appending new data to the list
        json_file.append(data)

        # Writing the updated list back to the JSON file
        with open(path, "w") as fp:
            json.dump(json_file, fp)
            fp.write("\n")
    print(f"Data saved to {path}")


def metrics_by_epoch(history: Dict) -> Dict:
    """Get the metrics by epoch.

    Args:
        history (Dict): history of the model.

    Returns:
        Dict: metrics by epoch.
    """
    metrics = {}
    for epoch, (loss, accuracy, val_loss, val_accuracy) in enumerate(
        zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        )
    ):
        metrics[epoch] = {
            "train_loss": loss,
            "train_accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    return metrics


if __name__ == "__main__":
    import doctest

    doctest.testmod()
