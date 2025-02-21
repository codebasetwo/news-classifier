import pytest
from newsfeed import train
import utilities

@pytest.mark.skip(reason="Compute intensive resource limitation")
@pytest.mark.training
def test_train_func(dataset_loc):
    experiment_name = utilities.generate_experiment_name()
    params =  '{ "num_epochs": 1, "batch_size": 32, "max_length": 128, "learning_rate": 5e-5,"num_classes": 6, "model": "bert-based-uncased", "num_samples": 250}'
    
    history = train.train_func(
        dataset_loc=dataset_loc,
        experiment_name=experiment_name,
        params=params,
        history_fp = None,
        split_train = True,
    )

    utilities.delete_experiment(experiment_name)
    assert history["val_loss"][-1] < history["val_loss"][-2]