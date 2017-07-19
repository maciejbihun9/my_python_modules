
from numpy import *

class Normalizer:

    @staticmethod
    def min_max_norm(data: ndarray, cols_to_norm: list) -> ndarray:
        """
        Returns data normalized between 0 and 1
        :param data: Data as ndarray to normalize
        :param cols_to_norm: list with columns to normalize. If empty then normalize all data.
        :return: normalized data as ndarray
        """
        m, n = shape(data)
        for col_to_norm in cols_to_norm:
            if col_to_norm in range(min(cols_to_norm), max(cols_to_norm)):
                print("Column index not in range: {}".format(col_to_norm))
                continue
            for i in range(m):
                data[i, col_to_norm] = (data[i, col_to_norm] - min(data[:, col_to_norm])) / (max(data[:, col_to_norm]) - min(data[:, col_to_norm]))
        return data