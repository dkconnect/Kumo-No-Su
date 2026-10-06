import matplotlib.pyplot as plt
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

degrees = np.arange(
    1,
    layer.degree + 1
)

powers = (
    x[:, None] ** degrees
)

contributions = np.sum(
    coefficients * powers,
    axis=1
)

bias = layer.b[
    HIDDEN_NODE
]

reconstructed = (
    bias
    + np.sum(contributions)
)

contribution_image = (
    contributions.reshape(
        28,
        28
    )
)

limit = np.max(
    np.abs(
        contribution_image
    )
)

print(
    "\nHidden node:",
    HIDDEN_NODE
)

print(
    "Bias:",
    bias
)

print(
    "Edge contribution sum:",
    np.sum(contributions)
)

print(
    "Hidden activation:",
    hidden[HIDDEN_NODE]
)

print(
    "Reconstructed:",
    reconstructed
)

print(
    "Difference:",
    abs(
        hidden[HIDDEN_NODE]
        - reconstructed
    )
)

plt.imshow(
    contribution_image,
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(
    label="Edge contribution"
)

plt.title(
    f"Kumo hidden node {HIDDEN_NODE} contributions"
)

plt.axis(
    "off"
)

plt.show()