"""
Implement the kmeans algorithm from scratch.
"""
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import random
from collections import defaultdict

from my_toolbox.mathematics.distance import euclidean_distance


class HomemadeKmeans:
    def __init__(self, k: Optional[int] = None):
        """
        k is the number of clusters, either user-defined or estimated
        """
        if k is not None:
            self.k = k
        self.centroids = []

    def fit(
        self,
        x_train: np.ndarray or pd.DataFrame,
        y_train: Optional[np.ndarray or pd.DataFrame] = None,
        threshold: float = 0.1,
        method: Optional[str] = None,
    ) -> Dict[int, List[int or float]]:
        """
        Kmeans algorithm is composed of two steps:
            - Expectation: Assign each data point to it's nearest centroid
            - Maximization: Update centroid based on new clusters

        Parameters
        --------
        x_train: np.ndarray or pd.DataFrame
            Training data
         y_train: Optional[np.ndarray or pd.DataFrame]
            Training data labels
        threshold: float
            If the difference between new and old centroids were less than the
            threshold, end the process
        """
        self.x_train = x_train
        self.y_train = y_train

        # Randomly select 3 points as centroids
        self.centroids = self._pick_init_centroids(method)
        prev_centroids = self.centroids
        self._expectation()
        self._maximization()
        while (
            np.mean(
                np.abs(
                    [
                        np.array(cur_cntr) - np.array(prv_cntr)
                        for cur_cntr, prv_cntr in zip(self.centroids, prev_centroids)
                    ]
                )
            )
            > threshold
        ):
            prev_centroids = self.centroids
            self._expectation()
            self._maximization()
        return self.clusters

    def _pick_init_centroids(self, method: Optional[str] = None) -> List[float or int]:
        """
        Pick starting centroids. If no method is specified, just randomly pick
        3 points
        """
        if method is None:
            self._init_centroids = random.sample(self.x_train.tolist(), self.k)
            self.centroids = self._init_centroids
            return self.centroids

    def _expectation(self) -> None:
        """
        Assign data to each centroids
        Use index as key
        """
        if len(self.centroids) == 0:
            raise ValueError("No centroids were picked.")
        self.clusters = defaultdict(list)
        for x_tr in self.x_train:
            assigned_centroid_idx = np.argmin(
                np.array([euclidean_distance(x_tr, cntr) for cntr in self.centroids])
            )
            self.clusters[assigned_centroid_idx].append(tuple(x_tr))

    def _maximization(self) -> None:
        """
        Find new centroids
        """
        self.centroids = [
            tuple(np.mean(points, axis=0)) for _, points in self.clusters.items()
        ]
