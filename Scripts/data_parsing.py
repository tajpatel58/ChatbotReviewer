# Import packages
import json
import pandas as pd
from pathlib import Path
from prefect import task, flow
from Scripts.model_params import ModelParams as Const
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


@flow
def load_data_flow(reviews_path: str):
    data_dict = load_json_data(reviews_path)
    reviews_df = parse_json_data_to_dataframe(data_dict)
    return reviews_df


if __name__ == "__main__":
    data_home = Path("/Users/tajsmac/Documents/Sentiment-Analysis/data")
    reviews_path = data_home / "reviews.json"
