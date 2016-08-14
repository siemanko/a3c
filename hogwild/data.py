import numpy as np

from sklearn.datasets import fetch_mldata

class Data(object):
    def __init__(self, batch_size, validation_size):
        self.batch_size = batch_size

        # Load MNIST
        mnist = fetch_mldata('MNIST original')
        X, Y_labels = mnist['data'], mnist['target']

        # normalize X to (0.0, 1.0) range
        X = X.astype(np.float32) / 255.0

        # one hot encode the labels
        Y = np.zeros((len(Y_labels), 10))
        Y[range(len(Y_labels)), Y_labels.astype(np.int32)] = 1.

        # ensure type is float32
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        # shuffle examples
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        Y = Y[permutation]

        # split into train, validate, test
        train_end      = 60000 - validation_size
        validation_end = 60000
        test_end       = 70000

        self.X_train = X[0:train_end]
        self.X_valid = X[train_end:validation_end]
        self.X_test  = X[validation_end:test_end]

        self.Y_train = Y[0:train_end]
        self.Y_valid = Y[train_end:validation_end]
        self.Y_test  = Y[validation_end:test_end]

    def iterate_batches(self, data_x, data_y):
        assert len(data_x) == len(data_y)

        for batch_start in range(0, len(data_x), self.batch_size):
            batch_x = data_x[batch_start:(batch_start + self.batch_size)]
            batch_y = data_y[batch_start:(batch_start + self.batch_size)]

            yield batch_x, batch_y

    def iterate_train(self):
        return self.iterate_batches(self.X_train, self.Y_train)

    def iterate_validate(self):
        return self.iterate_batches(self.X_valid, self.Y_valid)

    def iterate_test(self):
        return self.iterate_batches(self.X_test, self.Y_test)
