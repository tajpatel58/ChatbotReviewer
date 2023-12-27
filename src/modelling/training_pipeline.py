from pathlib import Path
from prefect import flow
from src.modelling.data_parsing import load_data_flow
from src.modelling.data_preprocessing import data_preprocessing_flow
from src.modelling.feature_engineering import feature_engineering_training_flow


@flow
def training_flow():
    project_home = Path("/Users/tajsmac/Documents/Sentiment-Analysis/")
    pickle_path = project_home / "src" / "pickles" / "preprocessing.pkl"
    json_reviews_path = project_home / "data" / "reviews.json"
    number_of_components = 40

    raw_reviews_data = load_data_flow(json_reviews_path)
    processed_data = data_preprocessing_flow(raw_reviews_data)
    feature_matrix, labels = feature_engineering_training_flow(
        processed_data, pickle_path, n_components=number_of_components
    )


if __name__ == "__main__":
    training_flow()
