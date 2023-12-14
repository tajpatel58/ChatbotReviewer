from pathlib import Path
from src.modelling.data_parsing import load_data_flow
from src.modelling.data_preprocessing import data_preprocessing_flow
from src.modelling.feature_engineering import feature_engineering_flow


data_home = Path("/Users/tajsmac/Documents/Sentiment-Analysis/")
json_reviews_path = data_home / "data" / "reviews.json"

raw_reviews_data = load_data_flow(json_reviews_path)
processed_data = data_preprocessing_flow(raw_reviews_data)
feature_matrix = feature_engineering_flow(processed_data)
