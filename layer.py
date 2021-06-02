class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.layer_size = len(neurons)

    def get_neurons(self):
        return self.neurons

    def get_neuron_weights(self):
        return [neuron.get_input_weights() for neuron in self.neurons]


class HiddenLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)


class OutputLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)
