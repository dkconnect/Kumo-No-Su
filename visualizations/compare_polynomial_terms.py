import numpy as np

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork


MODEL_PATH = "models/mnist_full_kumo.npz"
HIDDEN_NODE = 5

X_train, y_train, X_test, y_test = (
    load_mnist_split()
)

network = KumoNetwork.load(
    MODEL_PATH
)

layer = network.layers[0]

coefficients = layer.C[
    :,
    HIDDEN_NODE,
    :
]

bias = layer.b[
    HIDDEN_NODE
]

c1 = coefficients[:, 0]
c2 = coefficients[:, 1]

print(
    "\nHidden node:",
    HIDDEN_NODE
)

print(
    "Node bias:",
    bias
)

print(
    "\nDigit | Bias     | c1*x     | "
    "c2*x^2   | Activation"
)

print(
    "--------------------------------"
    "-------------------"
)

for digit in range(10):
    index = np.where(
        y_test == digit
    )[0][0]

    x = X_test[index]

    linear = np.sum(
        c1 * x
    )

    quadratic = np.sum(
        c2 * x ** 2
    )

    activation = (
        bias
        + linear
        + quadratic
    )

    real_activation = layer.forward(
        x
    )[HIDDEN_NODE]

    print(
        f"{digit:5d} | "
        f"{bias:8.4f} | "
        f"{linear:8.4f} | "
        f"{quadratic:8.4f} | "
        f"{activation:10.4f}"
    )

    if not np.isclose(
        activation,
        real_activation
    ):
        raise RuntimeError(
            "Polynomial decomposition "
            "does not match KumoLayer.forward()"
        )