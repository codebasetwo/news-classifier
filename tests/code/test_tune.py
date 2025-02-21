import pytest
from newsfeed import tune
import utilities


@pytest.mark.skip(reason = "assert logic not fully developed")
@pytest.mark.training
def test_tune_hyperparameters(dataset_loc):
    experiment_name = utilities.generate_experiment_name()
    best = tune.tune_hyperparameters(
        dataset_loc = dataset_loc,
        experiment_name = experiment_name,
    )
    pass
