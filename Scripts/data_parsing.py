# Import packages:
import re
import json
import pandas as pd
from prefect import task, flow
from Scripts.modelParams import ModelParams as Const
import nltk
from nltk.corpus import stopwords
import spacy
#nltk.download("stopwords")

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


def remove_punctuation(review_str : str) -> str:
    no_punct_str = re.sub('[^a-zA-Z]', ' ', review_str)
    return no_punct_str


def remove_stop_words(review_str : str) -> list:
    english_stop_words = set(stopwords.words("english"))
    review_list = re.findall(r'\S+', review_str)
    no_stop_words = [word.lower() for word in review_list if (word not in english_stop_words)]
    no_stop_words_str = " ".join(no_stop_words)
    return no_stop_words_str


def vectorize(review_str : str, nlp_vec):
    return nlp_vec(review_str)

if __name__ == "__main__":
    load_data_flow()
    