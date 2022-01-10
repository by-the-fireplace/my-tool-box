"""
Contains common scoring functions for ML models
"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np


class Scorer:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

    def print_metrics(self):
        print(
            f"---------- Metrics ----------\n"
            f"ROC AUC: {self.roc_auc}\n"
            f"F1: {self.f1}\n"
            f"Precision: {self.precision}\n"
            f"Recall: {self.recall}\n"
            f"Accuracy: {self.accuracy}"
            f"-----------------------------\n"
        )

    @property
    def roc_auc(self) -> float:
        return roc_auc_score(self.y_true, self.y_pred)

    @property
    def f1(self) -> float:
        return f1_score(self.y_true, self.y_pred)

    @property
    def precision(self) -> float:
        return precision_score(self.y_true, self.y_pred)

    @property
    def recall(self) -> float:
        return recall_score(self.y_true, self.y_pred)

    @property
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)
