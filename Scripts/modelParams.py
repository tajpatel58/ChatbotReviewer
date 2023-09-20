import os

class ModelParams(object):
    app_home_dir = "/Users/tajsmac/Documents/Sentiment-Analysis"
    data_relative_path = "data/reviews.json"
    data_path = os.path.join(app_home_dir, data_relative_path)

    REVIEW = "Review"
    LABEL_RAW = "Sentiment"
    REVIEW_WORDS = "Only_Words"
    REVIEW_NO_STOP_WORDS = "Review_No_Stopwords"
    BINARY_LABEL = "Binary_Sentiment"