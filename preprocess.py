import random


class PreProcess:
    @staticmethod
    def train_test_split(dataset, split=0.8):
        dataset_size = len(dataset)
        train_dataset_size = int(len(dataset) * split)
        test_dataset_size = dataset_size - train_dataset_size
        random.seed(10)
        test_dataset_index_list = random.sample(range(dataset_size), test_dataset_size)
        train_dataset, test_dataset = [], []
        for i, elem in enumerate(dataset):
            if i in test_dataset_index_list:
                test_dataset.append(elem)
            else:
                train_dataset.append(elem)
        return train_dataset, test_dataset

