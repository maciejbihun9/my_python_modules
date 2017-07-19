
import unittest
from src.normalizer import Normalizer
from src.data_manager import DataManager
from numpy import *

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
        data_narmalized = Normalizer.min_max_norm(self.data.astype(float), [4])
        print(data_narmalized)
