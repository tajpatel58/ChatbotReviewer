# Import packages
import os
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


def remove_punctuation(review_str : str) -> str:
    no_punct_str = re.sub('[^a-zA-Z]', ' ', review_str)
    return no_punct_str


def gen_embedding(review_str : str, nlp_lang):
    return nlp_lang(review_str).vector


def remove_stop_words(review_str : str) -> list:
    english_stop_words = set(stopwords.words("english"))
    review_list = re.findall(r'\S+', review_str)
    no_stop_words = [word.lower() for word in review_list if (word not in english_stop_words)]
    no_stop_words_str = " ".join(no_stop_words)
    return no_stop_words_str


@flow
def clean_reviews(reviews_df):
    reviews_df[Const.REVIEW_WORDS] = reviews_df[Const.REVIEW].apply(remove_punctuation)
    reviews_df[Const.REVIEW_NO_STOP_WORDS] = reviews_df[Const.REVIEW_WORDS].apply(remove_stop_words)
    label_mapping = {"positive_reviews" : 1, "negative_reviews" : 0, "neutral/constructive_reviews" : 0}
    reviews_df[Const.BINARY_LABEL] = reviews_df[Const.LABEL_RAW].replace(label_mapping)
    return reviews_df


@flow
def load_data_into_df(reviews_path : str):
    data_dict = load_json_data(reviews_path)
    reviews_df = parse_json_data_to_dataframe(data_dict)
    return reviews_df


@flow
def create_word_embeddings_mat(cleaned_reviews_df):
    nlp = spacy.load("en_core_web_md")
    embeddings_mat = pd.DataFrame(cleaned_reviews_df.apply
                              (lambda row : nlp(row[Const.REVIEW_NO_STOP_WORDS]).vector, 
                               axis=1).tolist())

    embeddings_mat[Const.BINARY_LABEL] = cleaned_reviews_df[Const.BINARY_LABEL].to_numpy()
    return embeddings_mat


@flow
def data_parsing_flow(reviews_path : str):
    reviews_df = load_data_into_df(reviews_path)
    cleaned_df = clean_reviews(reviews_df)
    embeddings_mat = create_word_embeddings_mat(cleaned_df)
    return embeddings_mat


if __name__ == "__main__":
    data_home = "/Users/tajsmac/Documents/Sentiment-Analysis/data"
    reviews_path = os.path.join(data_home, "reviews.json")
    data_parsing_flow(reviews_path)
    