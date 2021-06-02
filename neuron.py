import random


class Neuron:
    def __init__(self, input_weight_count, activation_function):
        self.input_weights = [random.random() for _ in range(input_weight_count)]
        self.output = None
        self.activation_function = activation_function

    def get_input_weights(self):
        return self.input_weights

    def get_output(self):
        return self.output

    def calculate_output(self, inputs, is_bias_present):
        activation = self.input_weights[-1] if is_bias_present else 0
        for i in range(len(self.input_weights) - 1):
            activation += self.input_weights[i] * inputs[i]
        output = self.activation_function.activate(activation)
        self.output = output
        return output
