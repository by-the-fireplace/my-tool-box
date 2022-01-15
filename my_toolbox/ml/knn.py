"""
Implement kNN algorithm from scratch
"""
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
from numpy import ndarray

from my_toolbox.mathematics.distance import euclidean_distance


def count_class(idx_class_pair: List[Any]) -> List[Tuple[Any, int]]:
    """
    Examples
    --------
    >>> l = ['a', 'a', 'b']
    >>> count_class(l)
    {'a': 2, 'b': 1}
    """
    res = defaultdict(int)
    for _class in idx_class_pair:
        res[_class] += 1
    return [(k, v) for k, v in res.items()]


class HomemadeKNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.y_pred = []

    def fit(self, x_train: ndarray, y_train: ndarray) -> None:
        """
        Nothing needs to be done here.
        """
        self.x_train = x_train
        self.y_train = y_train
        assert (
            self.k <= self.x_train.shape[0]
        ), "k cannot be greater than the number of data"

    def predict(self, x_test: ndarray) -> ndarray:
        """
        Steps:
        1. Calculate distance of all test data from every training data
        2. Sort distance
        3. Get top k nearest data
        4. Vote and find the most major class
        """

        for te in x_test:
            # Calculate distances
            # Use index as hashkeys
            distances = list(
                enumerate([euclidean_distance(tr, te) for tr in self.x_train])
            )
            # Sort distances
            distances.sort(key=lambda x: x[1])
            # Get top k classes
            top_k = distances[: self.k]
            top_k_class = [self.y_train[idx] for idx, _ in top_k]

            # Count and find major class
            class_count = count_class(top_k_class)
            sorted_class_l = sorted(class_count, reverse=True, key=lambda x: x[1])

            self.y_pred.append(sorted_class_l[0][0])
        return np.array(self.y_pred)
