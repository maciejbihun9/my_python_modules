
from numpy import *


class NormType:
    """
    Class with normalization types
    """

    @staticmethod
    def min_max_norm(data: ndarray, i: int, col_to_norm: int):
        data[i, col_to_norm] = (data[i, col_to_norm] - min(data[:, col_to_norm])) / (max(data[:, col_to_norm]) - min(data[:, col_to_norm]))

    @staticmethod
    def stand_norm(data: ndarray, i: int, col_to_norm: int):
        data[i, col_to_norm] = (data[i, col_to_norm] - mean(data[:, col_to_norm])) / std(data[:, col_to_norm])

    @staticmethod
    def unit_norm(data: ndarray, i: int, col_to_norm: int):
        data[i, col_to_norm] = (data[i, col_to_norm]) / max(data[:, col_to_norm])
