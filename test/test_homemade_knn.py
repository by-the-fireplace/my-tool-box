import numpy as np

from my_toolbox.ml.knn import HomemadeKNN


def test_homemade_knn():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]])
    y_train = np.array(["a", "a", "a", "b", "c"])
    x_test = np.array([[0.5, 0]])
    knn_classifier = HomemadeKNN(k=3)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    assert y_pred.tolist() == ["a"]
