import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    print("Loading MNIST...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto"
    )

    X = mnist.data.astype(
        np.float64
    )

    y = mnist.target.astype(
        np.int64
    )

    # normalizing pixels
    X = X / 255.0
    return X, y

if __name__ == "__main__":

    X, y = load_mnist()
    print("\nDataset loaded.")

    print(
        "Images shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    print(
        "\nFirst label:",
        y[0]
    )

    print(
        "First image shape:",
        X[0].shape
    )

    print(
        "Minimum pixel:",
        X[0].min()
    )

    print(
        "Maximum pixel:",
        X[0].max()
    )

    print(
        "\nFirst 20 pixels:"
    )

    print(
        X[0][:20]
    )