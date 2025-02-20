import pytest
from newsfeed import data


@pytest.fixture
def dataset_loc():
    return "../../datasets/train.csv"


@pytest.fixture
def processor():
    return data.DataPreprocessor()

