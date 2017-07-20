
import unittest
from src.normalizer import Normalizer
from src.data_manager import DataManager
from numpy import *
from src.math_oper import MathOper
from src.norm_type import NormType

class NormalizerTest(unittest.TestCase):

    def setUp(self):
        url = '../resources/german_data.txt'
        self.data_list = DataManager.load_data(url, False, False)
        categories = [True, False, True, True, False, True, True, False, True, True, False, True, False, True, True,
                      False, True, False, True, True]
        self.data = array([x[:19] for x in self.data_list])
        target = array([y[20] for y in self.data_list])

        self.target = [0 if y == '2' else 1 for y in target]
        DataManager.categorize_data(self.data, categories)

    def test_min_max_norm(self):
        data_normalized = Normalizer.normalize(self.data.astype(float), NormType.min_max_norm, [4])
        true_array = MathOper.values_between(data_normalized[:, 4], 0, 1)
        result = False in true_array
        self.assertFalse(result)

        data_normalized = Normalizer.normalize(self.data.astype(float), NormType.min_max_norm)
        true_array = MathOper.values_between(data_normalized, 0, 1)
        result = False in true_array
        self.assertFalse(result)

        data_normalized = Normalizer.normalize(self.data.astype(float), NormType.stand_norm)
        print(data_normalized)

        data_normalized = Normalizer.normalize(self.data.astype(float), NormType.unit_norm, [3])
        true_array = MathOper.values_between(data_normalized[:, 3], 0, 1)
        result = False in true_array
        self.assertFalse(result)

        data_normalized = Normalizer.normalize(self.data.astype(float), NormType.min_max_norm)
        true_array = MathOper.values_between(data_normalized, 0, 1)
        result = False in true_array
        self.assertFalse(result)



