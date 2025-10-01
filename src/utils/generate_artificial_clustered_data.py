from sklearn.datasets import make_blobs
import numpy as np


def generate_2D_clustered_data(
    N=1_000, n_clusters=3, seed=0, cluster_std=0.03, min_val=-1, max_val=1
):

    np.random.seed(seed)
    center_box = (min_val, max_val)
    X, y, centers = make_blobs(
        n_samples=N,
        centers=n_clusters,
        n_features=2,
        cluster_std=cluster_std,
        center_box=center_box,
        return_centers=True,
    )

    return X, y, centers


if __name__ == "__main__":
    X, y, centers = generate_2D_clustered_data(N=100, n_clusters=3, seed=1)
