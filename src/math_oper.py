
from numpy import *


class MathOper:

    @staticmethod
    def values_between(data_array: ndarray, min: float, max: float):
        return logical_and(data_array >= min, data_array <= max)