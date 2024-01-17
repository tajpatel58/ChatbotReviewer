import mlflow
import numpy as np
from pathlib import Path
from prefect import task, flow
from src.modelling import common
from src.modelling.model_params import ModelParams as Constants


@task(name="Fetching Model from Model Registry")
def fetch_model_from_registry_tasks():
    model_uri = f"models:/{Constants.MODEL_NAME}/latest/model"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return model


@flow(validate_parameters=False, log_prints=True)
def inference_flow(
    X: np.ndarray,
    mlflow_tracking_uri: Path,
):
    common.configure_mlflow_task(mlflow_tracking_uri)
    return None
