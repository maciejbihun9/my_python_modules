
import unittest
from src.data_manager import DataManager
from numpy import *

class DataManagerTest(unittest.TestCase):

    def setUp(self):
        url = '../resources/german_data.txt'
        self.data = DataManager.load_data(url, False, False)

    # TODO
    def test_load_data(self):
        return 0

    # TODO
    def test_categorize_data(self):
        inputs = array([x[:19] for x in self.data])
        categories = [True, False, True, True, False, True, True, False, True, True, False, True, False, True, True,
                      False, True, False, True, True]
        data_non_categoricals, data_categoricals = DataManager.categorize_data(inputs, categories)
        print(data_non_categoricals)
        print(data_categoricals)
        return 0
