import csv

from network import Network
from utils import Utils


class MLP:
    def __init__(self, hidden_layer_count, hidden_layer_size, activation_function, train_dataset_path, epochs=100,
                 learning_rate=0.3, bias_presence=None):
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = hidden_layer_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_dataset_path = train_dataset_path
        self.activation_function = activation_function
        self.train_dataset = None
        self.test_dataset = None
        self.bias_presence = True if bias_presence and bias_presence.lower() == 'yes' else False
        self.input_size = None
        self.output_size = None
        self.network = None

    def predict(self, inputs):
        outputs = self.network.forward_propagate(inputs)
        return outputs.index(max(outputs)) + 1

    def _read_train_dataset(self):
        with open(self.train_dataset_path, 'r') as csv_file:
            train_dataset_list = [row for row in csv.reader(csv_file)][1:]
        self.input_size = len(train_dataset_list[0]) - 1    # Subtract precision field, -1.
        self.output_size = len(set([row[-1] for row in train_dataset_list]))
        train_dataset_list = [[Utils.cast_float(row[0]), Utils.cast_float(row[1]), Utils.cast_int(row[-1])] for row in train_dataset_list]
        self.train_dataset = train_dataset_list

    def _read_test_dataset(self, test_dataset_path):
        with open(test_dataset_path, 'r') as csv_file:
            test_dataset_list = [row for row in csv.reader(csv_file)][1:]
        test_dataset_list = [[Utils.cast_float(row[0]), Utils.cast_float(row[1]), Utils.cast_int(row[-1])] for row in test_dataset_list]
        self.test_dataset = test_dataset_list

    def train(self):
        self._read_train_dataset()
        self.network = Network(self.hidden_layer_count, self.hidden_layer_size, self.bias_presence, self.input_size,
                               self.output_size, self.activation_function)
        for input_list in self.train_dataset:
            output = self.network.forward_propagate(input_list)
            # backpropagation
            # updating network with optimized weights
            print(output)

    def test(self, test_dataset_path):
        self._read_test_dataset(test_dataset_path)
        for row in self.test_dataset:
            expected = row[-1]
            prediction = self.predict(row[:-1])
            print(f'Expected: {expected}, Predicted: {prediction}')
