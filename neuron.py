import random


class Neuron:
    def __init__(self, input_weight_count, activation_function):
        self.input_weights = [random.random() - 0.5 for _ in range(input_weight_count)]
        self.output = None
        self.activation_function = activation_function
        self.delta = 0
        self.previous_delta = 0

    def get_input_weights(self):
        return self.input_weights

    def get_output(self):
        return self.output

    def get_delta(self):
        return self.delta

    def calculate_output(self, inputs, is_bias_present):
        activation = self.input_weights[-1] if is_bias_present else 0
        for input_index, input_weight in enumerate(self.input_weights[:-1] if is_bias_present else self.input_weights):
            activation += input_weight * inputs[input_index]
        output = self.activation_function.activate(activation)
        self.output = output
        return output
