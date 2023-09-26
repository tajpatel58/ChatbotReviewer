#%% 
import os
from Scripts.data_parsing import data_parsing_flow
from Scripts.featureEngineering import pca_reduce_flow

if __name__ == "__main__":
    data_home = "/Users/tajsmac/Documents/Sentiment-Analysis/data"
    reviews_path = os.path.join(data_home, "reviews.json")
    embed_mat = data_parsing_flow(reviews_path)
    reduced_features = pca_reduce_flow(embed_mat, n_comp=40)
    

# %%
