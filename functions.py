from abc import abstractmethod
from math import exp


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