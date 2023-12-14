import spacy
import pandas as pd
from prefect import task, flow
from sklearn.decomposition import PCA
from src.modelling.model_params import ModelParams as Const


@task
def pca_reduce_task(X: pd.DataFrame, n_comp: int):
    reduced_mat = PCA(n_components=n_comp, random_state=42).fit_transform(X.values)
    return reduced_mat


@task
def create_word_embeddings_mat_task(cleaned_reviews_df: pd.DatFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_md")
    embeddings_mat = cleaned_reviews_df[Const.REVIEW_NO_STOP_WORDS].apply(
        lambda x: nlp(x).vector
    )
    embeddings_mat[Const.BINARY_LABEL] = cleaned_reviews_df[
        Const.BINARY_LABEL
    ].to_numpy()
    return embeddings_mat


@flow
def feature_engineering_flow(cleaned_reviews_df: pd.DataFrame) -> pd.DataFrame:
    word_embeddings = create_word_embeddings_mat_task(cleaned_reviews_df)
    feature_matrix = pca_reduce_task(word_embeddings)
    return feature_matrix
