# Import packages
import json
import pandas as pd
from pathlib import Path
from prefect import task, flow
from src.modelling.model_params import ModelParams as Const
# nltk.download("stopwords")


@task
def load_json_data(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


@task
def parse_json_data_to_dataframe(data_dict: dict) -> pd.DataFrame:
    data_dfs = []
    for label, reviews_list in data_dict.items():
        sub_df = pd.DataFrame({Const.REVIEW: reviews_list})
        sub_df[Const.LABEL_RAW] = label
        data_dfs.append(sub_df)
    parsed_df = pd.concat(data_dfs, axis=0)
    return parsed_df


@flow(log_prints=True)
def load_data_flow(reviews_path: Path) -> pd.DataFrame:
    """
    Flow to load in training data (reviews data) from a json file. Json file should be only 1 key, 1 value layout.
    We need each key to be a class for our review: eg: "positive", and values should be a list of
    reviews belonging to that class.

    """
    data_dict = load_json_data(reviews_path)
    reviews_df = parse_json_data_to_dataframe(data_dict)
    print("Top 5 Rows Raw Data:")
    print(reviews_df.head())
    return reviews_df
