import great_expectations as gx
import pandas as pd
import pytest

def pytest_addoption(parser):
    """Add option to specify dataset location when executing tests from CLI.
    Ex: pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
    """
    parser.addoption("--dataset-loc", action="store", default=None, help="Dataset location.")


@pytest.fixture(scope="module")
def df(request):
    """Load dataset as a Great Expectations object."""
    dataset_loc = request.config.getoption("--dataset-loc")
    df = pd.read_csv(dataset_loc)
 
    return df

@pytest.fixture(scope="module")
def context():
    # Create a Data Context.
    context = gx.get_context()
    return context

