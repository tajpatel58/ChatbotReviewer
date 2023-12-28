import spacy
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
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
    """
    Parameters:
    -----------
    X (pd.DataFrame) : Dataframe on which we want to apply PCA onto.
    pca (sklearn.decomposition.PCA) : PCA object to fit/transform data.
    training (bool, defaults=True) : If false, we fit and transform pca, else only transform.


    Returns:
    --------
    reduced_mat (np.array) : Array of projected vectors.

    """
    if training:
        reduced_mat = pca.fit_transform(X.values)
    else:
        reduced_mat = pca.transform(X.values)
    return reduced_mat


@task(log_prints=True)
def load_pickled_objects_task(path_to_pickle: Path, log_msg: str) -> None:
    """
    Parameters:
    -----------
    path_to_pickle (pathlib.Path) : Path to pickle file.


    Returns:
    --------
    objs (object) : Python object loaded from pickle.
    """
    try:
        with open(path_to_pickle, "rb") as f:
            objs = pickle.load(f)
        print(log_msg)
    except FileNotFoundError:
        print(f"Pickle at path: {path_to_pickle} not found!")
    return objs


@task(log_prints=True)
def pickle_objects_task(
    path_to_pickle: Path,
    objs: object,
    log_msg: str,
) -> None:
    """
    Parameters:
    -----------
    path_to_pickle (pathlib.Path) : Path of where to save objects.
    objs (object) : Python object to pickle.
    log_msg (str) : Message to print to log console, should indicate what is being pickled.

    Returns:
    --------
    None

    """
    path_to_pickle.parent.mkdir(parents=True, exist_ok=True)
    print(log_msg)
    with open(path_to_pickle, "wb") as f:
        pickle.dump(objs, f)
    return None


@task(log_prints=True)
def transforming_labels_task(labels: pd.Series, label_encoder: LabelEncoder):
    """
    Groups "neutral/constructive" reviews as "negative" to turn problem into binary classification.
    Further fits/transforms a label encoder.

    Parameters:
    -----------
    labels (pd.Series) : Pandas series of training dataset labels.
    label_encoder (sklearn.preprocessing.LabelEncoder) : Label encoder instance to keep track of label mappings.

    Returns:
    --------
    mapped_labels (pd.Series) : Numerical labels for training data.

    """
    # group neutral reviews as negative:
    replace_dict = {"neutral/constructive": "negative"}
    replaced_labels = labels.replace(replace_dict)
    # transform labels:
    mapped_labels = label_encoder.fit_transform(replaced_labels)
    label_encoder_dict = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    print("Label Encoder Mapping is:")
    print(label_encoder_dict)
    return mapped_labels


@flow(validate_parameters=False, log_prints=True)
def feature_engineering_training_flow(
    cleaned_reviews_df: pd.DataFrame,
    path_to_pickle: str,
    n_components: int,
) -> tuple[np.array, np.array]:
    """
    Prefect flow, to carry out feature engineering on cleaned TRAINING data. The flow goes:
        - Generate word embeddings from string text.
        - Fit and transform PCA with a certain number of components
        - Transform training labels with a label encoder.
        - Pickle PCA and Encoder needed for inference later.

    Parameters:
    -----------
    cleaned_reviews_df (pd.DataFrame) : Dataframe with cleaned text data and a columns for labels.
    path_to_pickle (pathlib.Path) : Path of where to save preprocessing objects.
    n_components (int) : Number of components to be used when fitting PCA.

    Returns:
    --------
    pca_reduced_matrix (np.array) : Array of projected training data.
    labels (np.array) : Transformed training data labels.

    """
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

    pickle_objects_task(
        path_to_pickle=path_to_pickle,
        objs=preprocessing_dict,
        log_msg="pickled datapreprocessing objects",
    )

    print("Top 5 rows and last 5 columns of feature matrix:")
    print(pca_reduced_matrix[:5, -6:])
    return pca_reduced_matrix, labels


@flow(validate_parameters=False, log_prints=True)
def feature_engineering_inference_flow(
    cleaned_reviews_df: pd.DataFrame, path_to_pickle: str
):
    """
    Prefect flow, to carry out feature engineering on cleaned INFERENCE data. The flow goes:
        - Generate word embeddings from string text.
        - Load fitted pickle of PCA and encoder
        - Transform only - PCA with a certain number of components
        - No transformation of labels, as no labels at inference

    Parameters:
    -----------
    cleaned_reviews_df (pd.DataFrame) : Dataframe with cleaned text data and a columns for labels.
    path_to_pickle (pathlib.Path) : Path of where to save preprocessing objects.
    n_components (int) : Number of components to be used when fitting PCA.

    Returns:
    --------
    pca_reduced_matrix (np.array) : Array of projected inference data.

    """
    # load pickle file with preprocessing objects:
    preprocessing_dict = load_pickled_objects_task(
        path_to_pickle=path_to_pickle,
        log_msg="loading pickled dictionary of preprocessing objects",
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
