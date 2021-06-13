import csv
import math
from collections import defaultdict

import numpy as np

from functions import LinearFunction, SigmoidFunction, SoftmaxFunction
from network import Network
from preprocess import PreProcess
from utils import Utils


class MLP:
    def __init__(self, hidden_layer_count, hidden_layer_size, hidden_layer_activation_function,
                 output_layer_activation_function, train_dataset_path,
                 epochs=1, learning_rate=0.3, bias_presence=None, batch_size=None, epsilon=0.1, momentum=0):
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = hidden_layer_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_dataset_path = train_dataset_path
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_activation_function
        self.batch_size = batch_size
        self.train_dataset = None
        self.train_dataset_size = None
        self.validation_dataset = None
        self.training_score = None
        self.validation_score = None
        self.test_dataset = None
        self.bias_presence = True if bias_presence and bias_presence.lower() == 'yes' else False
        self.input_size = None
        self.network = None
        self.is_mnist_dataset = False
        self.epsilon = epsilon
        self.early_stop_max_count = 100
        self.momentum = momentum


class MLPClassification(MLP):
    def __init__(self, hidden_layer_count, hidden_layer_size, hidden_layer_activation_function, train_dataset_path,
                 epochs=1, learning_rate=0.3, bias_presence=None, batch_size=None, momentum=0):
        super().__init__(hidden_layer_count=hidden_layer_count, hidden_layer_size=hidden_layer_size,
                         hidden_layer_activation_function=hidden_layer_activation_function,
                         output_layer_activation_function=None,
                         train_dataset_path=train_dataset_path,
                         epochs=epochs, learning_rate=learning_rate, bias_presence=bias_presence,
                         batch_size=batch_size, momentum=momentum)

        self.output_size = None
        self.encoded_classes_key_real = defaultdict()
        self.encoded_classes_key_index = defaultdict()
        self.X = []
        self.y = []

    def _read_train_dataset(self):
        with open(self.train_dataset_path, 'r') as csv_file:
            csv_content = [row for row in csv.reader(csv_file)]
            csv_header = csv_content[0]
            if 'pixel' in csv_header[1].lower():
                self.is_mnist_dataset = True
            train_dataset_list = csv_content[1:]
        self.input_size = len(train_dataset_list[0]) - 1    # Subtract precision field, -1.
        output_classes = sorted(set([row[0] if self.is_mnist_dataset else row[-1] for row in train_dataset_list]))
        for index, output_class in enumerate(output_classes):
            self.encoded_classes_key_real[Utils.cast_int(output_class)] = index
            self.encoded_classes_key_index[index] = Utils.cast_int(output_class)
        self.output_size = len(output_classes)
        division_to_normalize = 255 if self.is_mnist_dataset else 1
        train_dataset_list = [[*[Utils.cast_float(i)/division_to_normalize for i in (row[1:] if self.is_mnist_dataset else row[:-1])], self.encoded_classes_key_real[Utils.cast_int(row[0] if self.is_mnist_dataset else row[-1])]] for row in train_dataset_list]
        # Train/validation split
        self.train_dataset, self.validation_dataset = PreProcess.train_test_split(train_dataset_list)
        self.train_dataset_size = len(self.train_dataset)
        self.batch_size = self.batch_size if self.batch_size else self.train_dataset_size
        if len(output_classes) == 2:
            self.output_layer_activation_function = SigmoidFunction()
        else:
            self.output_layer_activation_function = SoftmaxFunction()
        return {'real_to_index': self.encoded_classes_key_real, 'index_to_real': self.encoded_classes_key_index}

    def _read_test_dataset(self, test_dataset_path):
        with open(test_dataset_path, 'r') as csv_file:
            test_dataset_list = [row for row in csv.reader(csv_file)][1:]
        output_classes = sorted(set([row[-1] for row in test_dataset_list]))
        encoded_classes_key_real = defaultdict()
        encoded_classes_key_index = defaultdict()
        for index, output_class in enumerate(output_classes):
            encoded_classes_key_real[Utils.cast_int(output_class)] = index
            encoded_classes_key_index[index] = Utils.cast_int(output_class)
        test_dataset_list = [[*[Utils.cast_float(i) for i in row[:-1]], self.encoded_classes_key_real[Utils.cast_int(row[-1])]] for row in test_dataset_list]
        self.X = []
        self.y = []
        for i in test_dataset_list:
            self.y.append(i[-1])
            self.X.append(i[:-1])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.test_dataset = test_dataset_list
        return {'real_to_index': encoded_classes_key_real, 'index_to_real': encoded_classes_key_index}

    def _read_mnist_test_dataset(self, test_dataset_path):
        with open(test_dataset_path, 'r') as csv_file:
            test_dataset_list = [row for row in csv.reader(csv_file)][1:]
        test_dataset_list = [[*[Utils.cast_float(i)/255 for i in row]] for row in test_dataset_list]
        self.test_dataset = test_dataset_list
        return {'real_to_index': {i: i for i in range(10)}, 'index_to_real': {i: i for i in range(10)}}

    def train(self):
        training_score, validation_score = [], []
        encoded_classes = self._read_train_dataset()
        self.network = Network(self.hidden_layer_count, self.hidden_layer_size, self.bias_presence, self.input_size,
                               self.output_size, self.hidden_layer_activation_function,
                               self.output_layer_activation_function, self.momentum)
        num_of_iterations = math.floor(self.train_dataset_size / self.batch_size)
        prev_validation_metric = 0
        early_stopping_triggered_count = 0
        for epoch_num in range(self.epochs):
            total_error = 0
            for iteration in range(num_of_iterations):
                batch_dataset = self.train_dataset[iteration*self.batch_size:(iteration+1)*self.batch_size]
                input_list = None
                for input_index, input_list in enumerate(batch_dataset):
                    expected_encoded_class = input_list[-1]
                    input_list = input_list[:-1]
                    outputs = self.network.feed_forward(input_list)

                    expected = [0 for _ in range(self.output_size)]  # [0, 0]
                    expected[expected_encoded_class] = 1
                    total_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])

                    self.network.backward_propagate(expected)
                else:
                    self.network.update_weights(input_list, self.learning_rate)

            print(f"MSE: {total_error/self.train_dataset_size}, Epoch: {epoch_num}")

            validation_metric = self.evaluate(self.network, self.validation_dataset, encoded_classes)['accuracy']
            validation_score.append((epoch_num, validation_metric))
            diff = validation_metric - prev_validation_metric
            print(f"Accuracy: {'{:.2f}'.format(validation_metric)}%, "
                  f"Previous accuracy: {'{:.2f}'.format(prev_validation_metric)}%, "
                  f"Diff: {diff}, "
                  f"Epoch: {epoch_num}\n")
            if abs(diff) < self.epsilon:
                early_stopping_triggered_count += 1
            else:
                early_stopping_triggered_count = 0
            if early_stopping_triggered_count == self.early_stop_max_count:
                print(f"Early stopped after {epoch_num} epochs.")
                break
            prev_validation_metric = validation_metric
        return validation_score

    def test(self, test_dataset_path):
        if self.is_mnist_dataset:
            encoded_classes_dict = self._read_mnist_test_dataset(test_dataset_path)
            predictions = self.evaluate_mnist(self.network, self.test_dataset, encoded_classes_dict)
            with open("./datasets/MNIST/submission.csv", "w") as f:
                f.write("ImageId,Label\n")
                f.writelines([f"{i+1},{int(prediction)}\n" for i, prediction in enumerate(predictions)])
        else:
            encoded_classes_dict = self._read_test_dataset(test_dataset_path)
            test_metric = self.evaluate(self.network, self.test_dataset, encoded_classes_dict)['accuracy']
            print(f"Test Result: {test_metric}")
            return test_metric

    def predict_image(self, image_pixel_list):
        result = self.predict(self.network, image_pixel_list)
        return result

    def predict_last(self, array):
        return_z = []
        for row in array:
            outputs = self.network.feed_forward(row)
            return_z.append(outputs.index(max(outputs)))
        return return_z

    @staticmethod
    def predict(network, row):
        outputs = network.feed_forward(row)
        return outputs.index(max(outputs))

    @staticmethod
    def evaluate(network, test_set, encoded_classes):
        predictions = list()
        for row in test_set:
            prediction = MLPClassification.predict(network, row)
            if encoded_classes and (decoded_class := encoded_classes['index_to_real'].get(prediction)):
                prediction = decoded_class
            predictions.append(prediction)

        actual = [encoded_classes['index_to_real'][row[-1]] for row in test_set]

        return {
            'accuracy': MLPClassification.accuracy_metric(actual, predictions)
        }

    @staticmethod
    def evaluate_mnist(network, test_set, encoded_classes):
        predictions = list()
        for row in test_set:
            prediction = MLPClassification.predict(network, row)
            if encoded_classes and (decoded_class := encoded_classes['index_to_real'].get(prediction)):
                prediction = decoded_class
            predictions.append(prediction)

        return predictions

    @staticmethod
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


class MLPRegression(MLP):
    def __init__(self, hidden_layer_count, hidden_layer_size, hidden_layer_activation_function, train_dataset_path,
                 epochs=1, learning_rate=0.3, bias_presence=None, batch_size=None, momentum=0):
        super().__init__(hidden_layer_count=hidden_layer_count, hidden_layer_size=hidden_layer_size,
                         hidden_layer_activation_function=hidden_layer_activation_function,
                         output_layer_activation_function=LinearFunction(),
                         train_dataset_path=train_dataset_path,
                         epochs=epochs, learning_rate=learning_rate, bias_presence=bias_presence,
                         batch_size=batch_size, momentum=momentum)

        self.output_size = 1

    def _read_train_dataset(self):
        with open(self.train_dataset_path, 'r') as csv_file:
            train_dataset_list = [row for row in csv.reader(csv_file)][1:]
        self.input_size = len(train_dataset_list[0]) - 1    # Subtract precision field, -1.
        train_dataset_list = [[*[Utils.cast_float(i) for i in row[:-1]], Utils.cast_float(row[-1])] for row in train_dataset_list]
        # Train/validation split
        self.train_dataset, self.validation_dataset = PreProcess.train_test_split(train_dataset_list)
        self.train_dataset_size = len(self.train_dataset)
        self.batch_size = self.batch_size if self.batch_size else self.train_dataset_size

    def _read_test_dataset(self, test_dataset_path):
        with open(test_dataset_path, 'r') as csv_file:
            test_dataset_list = [row for row in csv.reader(csv_file)][1:]
        self.input_size = len(test_dataset_list[0]) - 1  # Subtract precision field, -1.
        test_dataset_list = [[*[Utils.cast_float(i) for i in row[:-1]], Utils.cast_float(row[-1])] for row in test_dataset_list]
        self.X = []
        self.y = []
        for i in test_dataset_list:
            self.y.append(i[-1])
            self.X.append(i[0])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.test_dataset = test_dataset_list

    def train(self):
        training_score, validation_score = [], []
        self._read_train_dataset()
        self.network = Network(self.hidden_layer_count, self.hidden_layer_size, self.bias_presence, self.input_size,
                               self.output_size, self.hidden_layer_activation_function,
                               self.output_layer_activation_function, self.momentum)
        num_of_iterations = math.floor(self.train_dataset_size / self.batch_size)
        prev_validation_metric = 0
        early_stopping_triggered_count = 0
        for epoch_num in range(self.epochs):
            total_error = 0
            for iteration in range(num_of_iterations):
                batch_dataset = self.train_dataset[iteration*self.batch_size:(iteration+1)*self.batch_size]
                input_list = None
                for input_index, input_list in enumerate(batch_dataset):
                    expected = input_list[-1]
                    input_list = input_list[:-1]
                    outputs = self.network.feed_forward(input_list)
                    total_error += (expected - outputs[0]) ** 2

                    self.network.backward_propagate([expected])
                else:
                    self.network.update_weights(input_list, self.learning_rate)

            validation_metric = self.evaluate(self.network, self.validation_dataset)['MSE']
            validation_score.append((epoch_num, validation_metric))
            diff = validation_metric - prev_validation_metric
            print(f"Validation MSE: {'{:.6f}'.format(validation_metric)}, "
                  f"Previous MSE: {'{:.6f}'.format(prev_validation_metric)}, "
                  f"Diff: {diff}, "
                  f"Epoch: {epoch_num}\n")
            if abs(diff) < self.epsilon:
                early_stopping_triggered_count += 1
            else:
                early_stopping_triggered_count = 0
            if early_stopping_triggered_count == self.early_stop_max_count:
                print(f"Early stopped after {epoch_num} epochs.")
                break
            prev_validation_metric = validation_metric
        return validation_score

    def test(self, test_dataset_path):
        self._read_test_dataset(test_dataset_path)
        test_metric = self.evaluate(self.network, self.test_dataset)['MSE']
        print(f"Test Result: {test_metric}")
        return test_metric

    @staticmethod
    def predict(network, row):
        outputs = network.feed_forward(row)

        return outputs[0]

    @staticmethod
    def evaluate(network, test_set):
        predictions = list()
        for row in test_set:
            prediction = MLPRegression.predict(network, row)
            predictions.append(prediction)

        actual = [row[-1] for row in test_set]

        return {
            'MSE': MLPRegression.mse_metric(actual, predictions)
        }

    @staticmethod
    def mse_metric(actual, predicted):
        total_error = 0
        for i in range(len(actual)):
            total_error += (actual[i] - predicted[i]) ** 2
        return total_error / float(len(actual))
