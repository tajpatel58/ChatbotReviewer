from prefect import task, flow
from sklearn.decomposition import PCA

@flow
def pca_reduce_flow(X, n_comp : int):
    reduced_mat = PCA(n_components=n_comp, random_state=42).fit_transform(X.values)
    return reduced_mat