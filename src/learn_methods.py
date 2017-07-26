from numpy import *

class LearnMethod:

    @staticmethod
    def adadelta():
        return 0

    @staticmethod
    def adagrad(alfa: float, cur_grad: ndarray, prev_grads: ndarray) -> ndarray:
        """
        Method should return ndarray of params delta.
        :param alfa: constant learning rate
        :param cur_grad: current gradient ndarray
        :param prev_grads: ndarray of previous gradients
        :return: ndarray of params delta
        """
        pow_grads = pow(prev_grads, 2)
        return -(alfa / (sqrt(pow_grads, axis = 0))) * cur_grad
