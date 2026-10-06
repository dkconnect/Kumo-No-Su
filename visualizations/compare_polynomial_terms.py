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

c0 = coefficients[:, 0]
c1 = coefficients[:, 1]
c2 = coefficients[:, 2]

constant_total = np.sum(c0)

print(
    "\nHidden node:",
    HIDDEN_NODE
)

print(
    "Constant baseline:",
    constant_total
)

print(
    "\nDigit | c0       | c1*x     | "
    "c2*x²    | Activation"
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

    constant = constant_total

    linear = np.sum(
        c1 * x
    )

    quadratic = np.sum(
        c2 * x ** 2
    )

    activation = (
        constant
        + linear
        + quadratic
    )

    print(
        f"{digit:5d} | "
        f"{constant:8.4f} | "
        f"{linear:8.4f} | "
        f"{quadratic:8.4f} | "
        f"{activation:10.4f}"
    )