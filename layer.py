class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.layer_size = len(neurons)
        self.is_hidden_layer = False
        self.is_output_layer = False
        self.previous_layer = None
        self.next_layer = None

    def get_neurons(self):
        return self.neurons

    def get_neuron_weights(self):
        return [neuron.get_input_weights() for neuron in self.neurons]

    def get_neuron_outputs(self):
        return [neuron.get_output() for neuron in self.neurons]

    def get_next_layer(self):
        return self.next_layer

    def get_previous_layer(self):
        return self.previous_layer


class HiddenLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)
        self.is_hidden_layer = True


class OutputLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)
        self.is_output_layer = True
