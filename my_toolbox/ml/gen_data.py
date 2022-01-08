# Different data generaters
from random import randrange

import numpy as np
from numpy import ndarray


def simple_2d(
    num_data: int, num_features: int, low: int or float = 0.0, high: int or float = 1.0
) -> ndarray:
    """
    Examples
    --------
    >>> simple_2d(3, 4, 1, 10)
    array([[2.65543147, 8.07816435, 1.77582935, 4.33328607],
           [9.17178329, 8.7913873 , 2.93240203, 5.6730138 ],
           [9.5141734 , 7.61628139, 3.74862091, 3.69017953]])
    """
    return np.random.rand(num_data, num_features) * (high - low) + low


def sample_class(size: int, pool: list) -> ndarray:
    """
    Examples
    --------
    >>> pool = ['a', 'b', 'c']
    >>> sample_class(10, pool)
    ['c', 'b', 'a', 'a', 'c', 'c', 'c', 'c', 'b', 'b']

    >>> pool = [0, 1]
    >>> sample_class(10, pool)
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 1]
    """
    res = []
    for i in range(size):
        rand_idx = randrange(len(pool))
        res.append(pool[rand_idx])
    return res
