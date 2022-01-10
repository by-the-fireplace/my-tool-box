import numpy as np

from my_toolbox.ml.kmeans import HomemadeKmeans


def test_homemade_kmeans():
    x_train = np.array([(1, 2), (1, 4), (1, 0), (100, 2), (100, 4), (100, 0)])
    kmeans = HomemadeKmeans(k=2)
    kmeans.fit(x_train, threshold=1e-10)
    print(kmeans._init_centroids)
    print(kmeans.centroids)
    print(kmeans.clusters)
    assert set(kmeans.centroids) == {(1, 2), (100, 2)}
    assert (
        (set(kmeans.clusters[0]) == {(1, 2), (1, 4), (1, 0)})
        and (set(kmeans.clusters[1]) == {(100, 2), (100, 4), (100, 0)})
    ) or (
        (set(kmeans.clusters[1]) == {(1, 2), (1, 4), (1, 0)})
        and (set(kmeans.clusters[0]) == {(100, 2), (100, 4), (100, 0)})
    )
