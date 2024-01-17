import mlflow
from pathlib import Path
from prefect import task
from src.modelling.model_params import ModelParams as Constants


@task(name="Configuring MlFlow")
def configure_mlflow_task(tracking_uri_path: Path) -> None:
    tracking_path = "sqlite:///" + str(tracking_uri_path)
    print(tracking_path)
    mlflow.set_tracking_uri(tracking_path)
    mlflow.set_experiment(Constants.EXPERIMENT_NAME)
    return None
