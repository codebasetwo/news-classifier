import os
import json
import pandas as pd
import pytest
from tempfile import TemporaryDirectory

from newsfeed import utils


@pytest.fixture(scope='function')
def df():
    return pd.DataFrame({'category': ['apple', 'banana', 'apple', 'orange']})


@pytest.fixture(scope='module')
def index_to_class():
    return {0: 'apple', 1: 'banana', 2: 'orange'}


@pytest.fixture(scope='module')
def class_to_index():
    return {'apple': 0, 'banana': 1, 'orange': 2}



@pytest.fixture(scope='module')
def metrics_epoch():
    return {0: {"train_loss":0.1,
                "train_accuracy":0.9,
                "val_loss":0.1,
                "val_accuracy":0.9},
            1:{"train_loss":0.2,
                "train_accuracy":0.8,
                "val_loss":0.2,
                "val_accuracy":0.8},
            }

def test_encode(df, class_to_index):
    encoded_mapping = utils.encode(df)
    assert encoded_mapping == class_to_index


def test_decode(index_to_class):
    indices = [0, 1, 2]
    indexed_class = utils.decode(indices, index_to_class)
    assert indexed_class == ['apple', 'banana', 'orange']


def test_save_dict_to_json_new_file():
    """Tests that the function saves a dictionary to a new file."""
    with TemporaryDirectory() as temp_dir:
        # Define the path and data
        path = os.path.join(temp_dir, "data.json")
        data = {"key": "value"}

        # Save the dictionary to a new file
        utils.save_dict_to_json(data, path)

        # Verify the file exists
        assert os.path.exists(path)

        # Verify the content of the file
        with open(path, "r") as fp:
            content = json.load(fp)
            assert content == [data]


def test_save_dict_to_json_existing_file():
    """Tests that the function appends data to an existing file."""
    with TemporaryDirectory() as temp_dir:
        # Define the path and initial data
        path = os.path.join(temp_dir, "data.json")
        initial_data = {"key1": "value1"}
        new_data = {"key2": "value2"}

        # Save initial data to the file
        with open(path, "w") as fp:
            json.dump([initial_data], fp)

        # Append new data to the existing file
        utils.save_dict_to_json(new_data, path)

        # Verify the content of the file
        with open(path, "r") as fp:
            content = json.load(fp)
            assert content == [initial_data, new_data]

def test_save_dict_to_json_create_directory():
    """Tests that the function creates a new directory if it does not exist."""
    with TemporaryDirectory() as temp_dir:
        # Define a path with a new directory
        new_dir = os.path.join(temp_dir, "new_dir")
        path = os.path.join(new_dir, "data.json")
        data = {"key": "value"}

        # Save the dictionary to a new file in a new directory
        utils.save_dict_to_json(data, path)

        # Verify the directory and file exist
        assert os.path.exists(new_dir)
        assert os.path.exists(path)

        # Verify the content of the file
        with open(path, "r") as fp:
            content = json.load(fp)
            assert content == [data]

def test_save_dict_to_json_sort_keys():
    """ Tests that the function sorts the keys of the dictionary before saving."""
    with TemporaryDirectory() as temp_dir:
        # Define the path and data
        path = os.path.join(temp_dir, "data.json")
        data = {"b": 2, "a": 1}

        # Save the dictionary with sorted keys
        utils.save_dict_to_json(data, path, sortkeys=True)

        # Verify the content of the file
        with open(path, "r") as fp:
            content = json.load(fp)
            assert content == [{"a": 1, "b": 2}]

def test_save_dict_to_json_custom_encoder():
    """ Tests that the function uses a custom encoder when provided."""
    with TemporaryDirectory() as temp_dir:
        # Define a custom encoder
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                return super().default(obj)

        # Define the path and data
        path = os.path.join(temp_dir, "data.json")
        data = {"key": {1, 2, 3}}

        # Save the dictionary with a custom encoder
        utils.save_dict_to_json(data, path, cls=CustomEncoder)

        # Verify the content of the file
        with open(path, "r") as fp:
            content = json.load(fp)
            assert content == [{"key": [1, 2, 3]}]


def test_metric_by_epoch(metrics_epoch):
    class History:
        def __init__(self):
            self.history = {'loss': [0.1, 0.2], 'accuracy': [0.9, 0.8], 'val_loss': [0.1, 0.2],
                            'val_accuracy': [0.9, 0.8]}
            
    metrics = utils.metrics_by_epoch(History())
    assert metrics == metrics_epoch




