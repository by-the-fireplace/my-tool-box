# Different types of distance
import numpy as np
from numpy import ndarray


def euclidean_distance(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Euclidean Distance:
    EuclideanDistance = sqrt(sum for i to N (v1[i] – v2[i])^2)

    Using numpy, there're 3 ways to calculate Euclidean distance

    1. Use np.linalg.norm()
    2. Use subtract and dot
    3. Use sum() and square()
    """
    # return np.sqrt(np.dot((x1 - x2).T, (x1 - x2)))
    # return np.linalg.norm(x1 - x2)
    return np.sqrt(np.sum(np.square(x1 - x2)))
