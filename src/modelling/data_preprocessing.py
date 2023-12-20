# Import packages
import re
import pandas as pd
from prefect import task, flow
from src.modelling.model_params import ModelParams as Const
from nltk.corpus import stopwords
# nltk.download("stopwords")


@task
def remove_punctuation_task(reviews_df: pd.DataFrame) -> pd.DataFrame:
    # Below regex, removes any non whitespace character that isn't a letter.
    reviews_df[Const.REVIEW_WORDS] = reviews_df[Const.REVIEW].str.replace(
        r"[^[a-zA-Z\s]", "", regex=True
    )
    return reviews_df


def remove_stop_words(review_str: str) -> str:
    english_stop_words = set(stopwords.words("english"))
    # Regex below finds sequences of non-whitespace characters.
    # (ie: words, but accounts for multiple spaces between words)
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
    return reviews_df


@task
def transform_classes_to_binary(reviews_df: pd.DataFrame) -> pd.DataFrame:
    label_to_bin_dict = {
        "positive": 1,
        "negative": 0,
        "neutral/constructive": 0,
    }
    reviews_df[Const.BINARY_LABEL] = reviews_df[Const.LABEL_RAW].map(label_to_bin_dict)
    return reviews_df


@flow(validate_parameters=False, log_prints=True)
def data_preprocessing_flow(raw_reviews_df: pd.DataFrame) -> pd.DataFrame:
    no_punctuation_df = remove_punctuation_task(raw_reviews_df)
    cleaned_df = remove_stop_words_task(no_punctuation_df)
    cleaned_with_binary_label_df = transform_classes_to_binary(cleaned_df)
    print("Top 5 Rows of Cleaned/Processed DF:")
    print(cleaned_with_binary_label_df.head())
    return cleaned_with_binary_label_df
