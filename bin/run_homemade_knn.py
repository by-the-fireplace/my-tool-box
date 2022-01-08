#!/usr/bin/env python

from my_toolbox.ml.gen_data import sample_class, simple_2d
from my_toolbox.ml.knn import HomemadeKNNClassifier
import matplotlib.pyplot as plt


def main() -> None:
    x_train = simple_2d(num_data=20, num_features=2, low=0, high=10)
    y_train = sample_class(size=20, pool=["a", "b", "c", "d"])
    x_test = simple_2d(num_data=5, num_features=2, low=0, high=10)
    knn_classifier = HomemadeKNNClassifier(k=3)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)

    # plotting
    color_map = {"a": "red", "b": "blue", "c": "black", "d": "green"}
    plt.scatter(
        x_train.T[0], x_train.T[1], color=[color_map[y_tr] for y_tr in y_train], s=150
    )
    plt.scatter(
        x_test.T[0],
        x_test.T[1],
        color=[color_map[y_pr] for y_pr in y_pred],
        s=50,
        marker="v",
    )
    plt.show()


if __name__ == "__main__":
    main()
