from pathlib import Path
from prefect import flow
from src.modelling.data_parsing import load_data_flow
from src.modelling.data_preprocessing import data_preprocessing_flow
from src.modelling.feature_engineering import feature_engineering_inference_flow
from src.modelling.inference import inference_flow


@flow
def inference_pipeline_flow():
    project_home = Path("/Users/taj/Documents/ChatbotReviewer/")
    pickle_path = project_home / "src" / "pickles" / "preprocessing.pkl"
    mlflow_uri_path = (
        project_home / "src" / "modelling" / "experiment_tracking" / "mlflow.db"
    )
    json_reviews_path = project_home / "data" / "reviews.json"

    raw_reviews_data = load_data_flow(json_reviews_path)
    processed_data = data_preprocessing_flow(raw_reviews_data)
    feature_matrix = feature_engineering_inference_flow(
        processed_data,
        pickle_path,
    )
    predictions = inference_flow(feature_matrix, mlflow_uri_path)
    return predictions


if __name__ == "__main__":
    inference_pipeline_flow()
