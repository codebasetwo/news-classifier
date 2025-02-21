import pandas as pd
import pytest
from newsfeed import predict


def pytest_addoption(parser):
    """Add option to specify name of experiment when executing tests from CLI.
    Ex: pytest --experiment-name=$EXPERIMENT_NAME tests/model --verbose --disable-warnings
    """
    parser.addoption("--experiment-name", action="store", default=None, help="name of your experiment")


@pytest.fixture(scope="module")
def run_id(request):
    """Load dataset as a Great Expectations object."""
    experiment_name = request.config.getoption("--experiment-name")
    run_id = predict.get_best_run_id(experiment_name)

    return run_id


@pytest.fixture(scope='module')
def model(run_id):
    return  predict.get_best_model(run_id)
