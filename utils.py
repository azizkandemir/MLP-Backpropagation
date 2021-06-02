from math import exp

import numpy as np


class Utils:
    @staticmethod
    def cast_int(s):
        try:
            return int(s)
        except:
            return None

    @staticmethod
    def cast_float(s):
        try:
            return float(s)
        except:
            return None

    @staticmethod
    def softmax(x):
        """
            https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
            Compute softmax values for each sets of scores in x.
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def sigmoid(x):
        """
            https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
            :param x:
            :return:
        """
        return 1.0 / (1.0 + exp(-x))

    @staticmethod
    def tanh(x):
        """
            https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
            :param x:
            :return:
        """
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
