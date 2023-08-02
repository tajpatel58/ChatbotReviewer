# Import packages:
import json
from prefect import task, flow
from Scripts.modelParams import ModelParams as Const

@task
def load_data(path : str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

@flow
def load_data_flow():
    load_data(Const.data_path)

if __name__ == "__main__":
    load_data_flow()