import datetime
from newsfeed.config import mlflow



def generate_experiment_name(prefix: str = "test_") -> str:
    date_str = datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    return f"{prefix}-{date_str}"


def delete_experiment(experiment_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)