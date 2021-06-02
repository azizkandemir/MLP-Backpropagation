import random

from layer import HiddenLayer, OutputLayer
from neuron import Neuron


class Network:
    def __init__(self, hidden_layer_count, hidden_layer_size, bias_presence, input_size, output_size, activation_function):
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_size = hidden_layer_size
        self.bias_presence = bias_presence
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = []
        self.activation_function = activation_function
        self.output_layer = None
        self.layers = None
        self._initialize_network()

    def _initialize_network(self):
        hidden_layer_count = self.hidden_layer_count
        hidden_layer_size = self.hidden_layer_size
        bias_presence = self.bias_presence
        input_size = self.input_size
        output_size = self.output_size
        activation_func = self.activation_function
        random.seed(10)
        for layer_index in range(hidden_layer_count):
            previous_layer_size = input_size if layer_index == 0 else hidden_layer_size
            neuron_list = [Neuron(previous_layer_size + (1 if bias_presence else 0), activation_func) for _ in range(hidden_layer_size)]
            hidden_layer = HiddenLayer(neuron_list)
            self.hidden_layers.append(hidden_layer)
        neuron_list = [Neuron(hidden_layer_size + (1 if bias_presence else 0), activation_func) for _ in range(output_size)]
        output_layer = OutputLayer(neuron_list)
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
        return inputs
