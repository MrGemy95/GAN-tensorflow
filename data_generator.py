from utils import load_mnist
import numpy as np


class DataGenerator():
    def __init__(self, config):
        self.config = config
        self.x_train, self.y_train = load_mnist(self.config.data_path, kind='train')
        self.x_test, self.y_test = load_mnist(self.config.data_path, kind='t10k')
        self.x_train = self.x_train.reshape(-1, 28, 28, 1)/255.0
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)/255.0

    def next_batch(self):
        while True:
            idx = np.random.choice(self.x_train.shape[0], self.config.batch_size)
            batch_x = self.x_train[idx]
            yield batch_x
