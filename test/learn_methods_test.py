
import unittest
from src.learn_methods import LearnMethod
import numpy as np



class LearnMetohdsTest(unittest.TestCase):

    def setUp(self):
        return 0


    def test_adagrad(self):
        alfa = 0.2
        cur = np.array([2,3,4])
        prev = np.array([1,5,2], [1,5,2], [1,5,2])
        delta = LearnMethod.adagrad(alfa, cur, prev)

