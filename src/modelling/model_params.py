import os


class ModelParams(object):
    app_home_dir = "/Users/tajsmac/Documents/Sentiment-Analysis"
    data_relative_path = "data/reviews.json"
    data_path = os.path.join(app_home_dir, data_relative_path)

    REVIEW = "review"
    LABEL_RAW = "sentiment"
    REVIEW_WORDS = "only_words"
    REVIEW_NO_STOP_WORDS = "review_no_stopwords_str"
    BINARY_LABEL = "binary_sentiment"
