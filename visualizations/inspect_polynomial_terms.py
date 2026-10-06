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

x = X_test[0]

activations = network.inspect(
    x
)

hidden = activations[1]

layer = network.layers[0]

coefficients = layer.C[
    :,
    HIDDEN_NODE,
    :
]

c0 = coefficients[:, 0]
c1 = coefficients[:, 1]
c2 = coefficients[:, 2]

constant_terms = c0
linear_terms = c1 * x
quadratic_terms = c2 * x ** 2

constant_total = np.sum(
    constant_terms
)

linear_total = np.sum(
    linear_terms
)

quadratic_total = np.sum(
    quadratic_terms
)

reconstructed = (
    constant_total
    + linear_total
    + quadratic_total
)

print("\nHidden node:", HIDDEN_NODE)

print(
    "Actual activation:",
    hidden[HIDDEN_NODE]
)

print("\nPolynomial term totals:")

print(
    "c0:     ",
    constant_total
)

print(
    "c1 * x: ",
    linear_total
)

print(
    "c2 * x²:",
    quadratic_total
)

print(
    "\nReconstructed:",
    reconstructed
)

print(
    "Difference:",
    abs(
        hidden[HIDDEN_NODE]
        - reconstructed
    )
)