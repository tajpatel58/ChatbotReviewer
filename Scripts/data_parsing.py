# Import packages:
import json
import pandas as pd
from prefect import task, flow
from Scripts.modelParams import ModelParams as Const

@task
def load_json_data(path : str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


@task
def parse_json_data_to_dataframe(data : dict) -> pd.DataFrame:
    data_dfs = []
    for label, reviews_list in data.items():
        sub_df = pd.DataFrame({Const.REVIEW : reviews_list})
        sub_df[Const.LABEL_RAW] = label
        data_dfs.append(sub_df)
    parsed_df = pd.concat(data_dfs, axis=0)
    return parsed_df


@flow
def load_data_flow():
    data_dict = load_json_data(Const.data_path)
    reviews_df = parse_json_data_to_dataframe(data_dict)


if __name__ == "__main__":
    load_data_flow()
    