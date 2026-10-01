from sklearn.datasets import fetch_openml
import numpy as np


def load_mnist():
    print("Loading MNIST")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto"
    )

    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(np.int64)
    X = X / 255.0
    return X, y


def load_mnist_split():

    X, y = load_mnist()
    X_train = X[:60000]
    y_train = y[:60000]

    X_test = X[60000:]
    y_test = y[60000:]

    return (X_train, y_train, X_test, y_test)