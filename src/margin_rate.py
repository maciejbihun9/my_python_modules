
from numpy import *

class MarginRate:

    @staticmethod
    def ada_delta(ro: float, grad: float, prev_avg: float, rate: float):
        """
        Instead of accumulating all squared gradients to t iteration.
        We get only average of the previous gradients. To compute our learning rate we use running average.
        E[g2]t = ρ E[g2]t−1 + (1 − ρ) g2
        :return:
        """
        # estimate running average of gradients
        epsilon = 2
        Eg_pow = ro * prev_avg + (1 - ro) * pow(grad, 2)
        rms = sqrt(Eg_pow + epsilon)
        return (rate / rms) * grad


    @staticmethod
    def momen_method(ro: float, prev_delta: float, gradient: float):
        """∆xt = ρ∆xt−1 − ηgt
        :return: New delta based on previous numbers
        """
        return ro * prev_delta - gradient

    @staticmethod
    def get_adagrad(rate: float, grad: float, prev_grads: ndarray):
        """
        This is specially helpful when working with neural nets because we end up
        with different values in each neuron. Neuron gradient value depends on previous values.
        Each layer has different values of learning rates.
        :param rate: Our selected learning rate value. It is stale value
        :param grad: Actual value of the gradient
        :param prev_grads: previous gradients
        :return:
        """
        return -(rate / sqrt(sum(pow(prev_grads, 2)))) * grad






"""
    

"""