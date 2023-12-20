import spacy
import pandas as pd
from prefect import task, flow
from sklearn.decomposition import PCA
from src.modelling.model_params import ModelParams as Const


@task(log_prints=True)
def create_word_embeddings_mat_task(
    cleaned_reviews_df: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    """
    Parameters:
    -----------
    cleaned_reviews_df (pd.DataFrame) : Dataframe with a column of cleaned text reviews.
    col_name (str) : String of col

    Returns:
    --------
    embeddings_mat (pd.DataFrame) : Dataframe of word embeddings, each column being a feature.

    Function to create a dataframe of word embeddings for cleaned text reviews.
    Word embeddings are generated using Spacy's pretrained model, and we expect each embedding
    to be of length 300.
    """
    word_embed_generator = spacy.load("en_core_web_md")
    embeddings_mat = cleaned_reviews_df[col_name].apply(
        lambda x: pd.Series(word_embed_generator(x).vector)
    )
    print("Embeddings Matrix Shape:")
    print(embeddings_mat.shape)
    return embeddings_mat


@task
def pca_reduce_task(X: pd.DataFrame, pca: PCA, inference=True):
    if inference:
        reduced_mat = pca.transform(X.values)
    else:
        reduced_mat = pca.fit_transform(X.values)
    return reduced_mat


@task
def load_fitted_pca_task(path_to_pickle: str):
    pass


@flow(validate_parameters=False, log_prints=True)
def feature_engineering_flow(
    cleaned_reviews_df: pd.DataFrame, inference: bool = True, n_components: int = None
) -> pd.DataFrame:
    if inference:
        path_to_pickle = " "
        pca = load_fitted_pca_task(path_to_pickle=path_to_pickle)
    else:
        pca = PCA(n_components=n_components, random_state=42)

    labels = cleaned_reviews_df[Const.BINARY_LABEL].values
    word_embeddings = create_word_embeddings_mat_task(
        cleaned_reviews_df, Const.REVIEW_NO_STOP_WORDS
    )
    feature_matrix = pca_reduce_task(word_embeddings, pca)
    print("Top 5 rows and last 5 columns of feature matrix:")
    print(feature_matrix[:5, -6:])
    return feature_matrix, labels
