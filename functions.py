from abc import abstractmethod
from math import exp

import numpy as np


class ActivationFunction:
    def __init__(self):
        pass

    @staticmethod
    def determine_function(func_str):
        return_function = None
        if func_str.lower() == 'sigmoid':
            return_function = SigmoidFunction()
        elif func_str.lower() == 'tanh':
            return_function = TanhFunction()
        elif func_str.lower() == 'softmax':
            return_function = SoftmaxFunction()

        return return_function

    @abstractmethod
    def activate(self, value):
        pass


class SigmoidFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, val):
        """
            https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
            :param val:
            :return:
        """
        return 1.0 / (1.0 + exp(-val))


class TanhFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, val):
        """
            https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
            :param val:
            :return:
                """
        return (exp(val) - exp(-val)) / (exp(val) + exp(-val))


class SoftmaxFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, val):
        """
            https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
            Compute softmax values for each sets of scores in x.
        """
        return np.exp(val) / np.sum(np.exp(val), axis=0)
