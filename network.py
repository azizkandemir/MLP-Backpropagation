import random

from functions import SoftmaxFunction
from layer import HiddenLayer, OutputLayer
from neuron import Neuron


class Network:
    def __init__(self, hidden_layer_count, hidden_layer_size, bias_presence, input_size, output_size,
                 hidden_layer_activation_function, output_layer_activation_function, momentum):
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_size = hidden_layer_size
        self.bias_presence = bias_presence
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = []
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_activation_function
        self.momentum = momentum
        self.output_layer = None
        self.layers = None
        self._initialize_network()

    def _initialize_network(self):
        hidden_layer_count = self.hidden_layer_count
        hidden_layer_size = self.hidden_layer_size
        bias_presence = self.bias_presence
        input_size = self.input_size
        output_size = self.output_size
        hidden_layer_activation_func = self.hidden_layer_activation_function
        output_layer_activation_func = self.output_layer_activation_function
        random.seed(10)
        for layer_index in range(hidden_layer_count):
            previous_layer_size = input_size if layer_index == 0 else hidden_layer_size
            neuron_list = [Neuron(previous_layer_size + (1 if bias_presence else 0), hidden_layer_activation_func) for _ in range(hidden_layer_size)]
            hidden_layer = HiddenLayer(neuron_list)
            hidden_layer.previous_layer = None if layer_index == 0 else self.hidden_layers[layer_index - 1]
            if hidden_layer.previous_layer:
                hidden_layer.previous_layer.next_layer = hidden_layer
            self.hidden_layers.append(hidden_layer)
        neuron_list = [Neuron(hidden_layer_size + (1 if bias_presence else 0), output_layer_activation_func) for _ in range(output_size)]
        output_layer = OutputLayer(neuron_list)
        self.hidden_layers[-1].next_layer = output_layer
        output_layer.previous_layer = self.hidden_layers[-1]
        self.output_layer = output_layer
        self.layers = [*self.hidden_layers, output_layer]

    def get_layers(self):
        return self.layers

    def forward_propagate(self, inputs):
        for layer in self.get_layers():
            new_inputs = []
            for neuron in layer.get_neurons():
                neuron.calculate_output(inputs, self.bias_presence)
                new_inputs.append(neuron.get_output())
            inputs = new_inputs
        if isinstance(self.output_layer_activation_function, SoftmaxFunction):
            inputs = list(SoftmaxFunction.activate_all(inputs))
        return inputs

    def backward_propagate(self, expected):
        """
        Backpropagate error and store in neurons
        :param expected:
        :return:
        """
        # From output layer to input layer
        # expected = [1,0]
        for layer in reversed(self.get_layers()):
            errors = []

            # For last layer just calculate errors
            if layer.is_output_layer:
                for neuron_index, neuron in enumerate(layer.get_neurons()):
                    errors.append(neuron.get_output() - expected[neuron_index])
            # For input and hidden layers
            else:
                for neuron_index in range(len(layer.get_neurons())):
                    error = 0.0
                    for next_layer_neuron in layer.get_next_layer().get_neurons():
                        error += (next_layer_neuron.get_input_weights()[neuron_index] * next_layer_neuron.get_delta())
                    errors.append(error)

            for neuron_index, neuron in enumerate(layer.get_neurons()):
                neuron.previous_delta = neuron.delta
                neuron.delta = errors[neuron_index] * neuron.activation_function.calculate_derivative(neuron.get_output())

    def update_weights(self, inputs, learning_rate):
        for layer_index, layer in enumerate(self.get_layers()):
            if layer_index != 0:
                inputs = layer.get_previous_layer().get_neuron_outputs()

            for neuron in layer.get_neurons():
                for input_index, input_data in enumerate(inputs):
                    neuron.input_weights[input_index] -= (learning_rate * neuron.get_delta() * input_data) + (self.momentum * neuron.previous_delta)
                if self.bias_presence:
                    neuron.input_weights[-1] -= learning_rate * neuron.get_delta() + (self.momentum * neuron.previous_delta)
