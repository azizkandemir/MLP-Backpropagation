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
        elif func_str.lower() == 'relu':
            return_function = ReLUFunction()
        elif func_str.lower() == 'linear':
            return_function = LinearFunction()
        elif func_str.lower() == 'softmax':
            return_function = SoftmaxFunction()
        return return_function

    @abstractmethod
    def activate(self, value):
        pass

    @abstractmethod
    def calculate_derivative(self, value):
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

    def calculate_derivative(self, val):
        return val * (1.0 - val)


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

    def calculate_derivative(self, val):
        return 1.0 - val ** 2


class ReLUFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, val):
        return max(0, val)

    def calculate_derivative(self, val):
        return 0 if val <= 0 else 1


class LinearFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, val):
        return val

    def calculate_derivative(self, val):
        return 1


class SoftmaxFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def activate(self, val):
        return val

    def calculate_derivative(self, value):
        return 1

    @staticmethod
    def activate_all(vals):
        """
            https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
        """
        return np.exp(vals) / np.sum(np.exp(vals), axis=0)
