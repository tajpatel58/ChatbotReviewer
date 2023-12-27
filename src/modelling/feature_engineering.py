import spacy
import pickle
import pandas as pd
from prefect import task, flow
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
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
def pca_reduce_task(X: pd.DataFrame, pca: PCA, training: bool = True) -> None:
    if training:
        reduced_mat = pca.fit_transform(X.values)
    else:
        reduced_mat = pca.transform(X.values)
    return reduced_mat


@task
def load_pickled_preprocessing_objects_task(path_to_pickle: str) -> None:
    try:
        with open(path_to_pickle, "rb") as f:
            pickle.load(f)
    except FileNotFoundError:
        print(f"Pickle at path: {path_to_pickle} not found!")
    return None


@task
def pickle_fitted_preprocessing_objects_task(
    path_to_pickle: str, preprocessing_objects: dict
) -> None:
    path_to_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_pickle, "wb") as f:
        pickle.dump(preprocessing_objects, f)
    return None


@task
def transforming_labels_task(labels: pd.Series, label_encoder: LabelEncoder):
    # group neutral reviews as negative:
    replace_dict = {"neutral/constructive": "negative"}
    replaced_labels = labels.map(replace_dict)
    # transform labels:
    labels = label_encoder.fit_transform(replaced_labels)
    return labels


@flow(validate_parameters=False, log_prints=True)
def feature_engineering_training_flow(
    cleaned_reviews_df: pd.DataFrame,
    path_to_pickle: str,
    n_components: int,
) -> pd.DataFrame:
    # initialise preprocessing objects:
    pca = PCA(n_components=n_components, random_state=42)
    label_encoder = LabelEncoder()

    # create word embeddings
    word_embeddings = create_word_embeddings_mat_task(
        cleaned_reviews_df, Const.REVIEW_NO_STOP_WORDS
    )

    # feature engineering/fitting preprocessing objects:
    pca_reduced_matrix = pca_reduce_task(word_embeddings, pca, training=True)
    labels = transforming_labels_task(
        cleaned_reviews_df[Const.LABEL_RAW], label_encoder=label_encoder
    )

    # save preprocessing components
    preprocessing_dict = {"pca": pca, "label_encoder": label_encoder}

    pickle_fitted_preprocessing_objects_task(
        path_to_pickle=path_to_pickle, preprocessing_objects=preprocessing_dict
    )

    print("Top 5 rows and last 5 columns of feature matrix:")
    print(pca_reduced_matrix[:5, -6:])
    return pca_reduced_matrix, labels


@flow(validate_parameters=False, log_prints=True)
def feature_engineering_inference_flow(
    cleaned_reviews_df: pd.DataFrame, path_to_pickle: str
):
    # load pickle file with preprocessing objects:
    preprocessing_dict = load_pickled_preprocessing_objects_task(
        path_to_pickle=path_to_pickle
    )

    # set preprocessng items:
    pca = preprocessing_dict["pca"]

    # create word embeddings
    word_embeddings = create_word_embeddings_mat_task(
        cleaned_reviews_df, Const.REVIEW_NO_STOP_WORDS
    )

    # feature engineering:
    pca_reduced_matrix = pca_reduce_task(word_embeddings, pca, training=False)
    print("Top 5 rows and last 5 columns of feature matrix:")
    print(pca_reduced_matrix[:5, -6:])
    return pca_reduced_matrix
