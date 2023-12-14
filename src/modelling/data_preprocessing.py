# Import packages
import re
import pandas as pd
from prefect import task, flow
from src.modelling.model_params import ModelParams as Const
from nltk.corpus import stopwords
# nltk.download("stopwords")


@task
def remove_punctuation_task(reviews_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df[Const.REVIEW_WORDS] = reviews_df[Const.REVIEW].str.replace(
        "[^a-zA-Z]", "", regex=True
    )
    return reviews_df


def remove_stop_words(review_str: str) -> str:
    english_stop_words = set(stopwords.words("english"))
    review_list = re.findall(r"\S+", review_str)
    no_stop_words = [
        word.lower() for word in review_list if (word not in english_stop_words)
    ]
    no_stop_words_str = " ".join(no_stop_words)
    return no_stop_words_str


@task
def remove_stop_words_task(reviews_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df[Const.REVIEW_NO_STOP_WORDS] = reviews_df[Const.REVIEW_WORDS].apply(
        remove_stop_words
    )


@flow
def data_preprocessing_flow(raw_reviews_df: pd.DataFrame) -> pd.DataFrame:
    no_punctuation_df = remove_punctuation_task(raw_reviews_df)
    cleaned_df = remove_stop_words_task(no_punctuation_df)
    return cleaned_df
