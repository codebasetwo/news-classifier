import numpy as np
import pytest

from newsfeed import predict

@pytest.fixture(scope='module')
def probabilities():
    return np.array([0.1, 0.7, 0.2])


@pytest.fixture(scope='module')
def index_to_class():
    return {0: "NEWS & POLITICS",  1: "ENTERTAINMENT", 2: "PARENTING"}


def test_format_probability(probabilities, index_to_class):
    value = predict.format_probability(probabilities, index_to_class)

    assert value == {"NEWS & POLITICS": 0.1, "ENTERTAINMENT": 0.7, "PARENTING": 0.2}
