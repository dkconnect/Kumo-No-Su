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


figure, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4)
)

axes[0].imshow(
    x.reshape(28, 28),
    cmap="gray"
)

axes[0].set_title(
    f"MNIST digit {y_test[0]}"
)

axes[0].axis(
    "off"
)

image = axes[1].imshow(
    contribution_image,
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit
)

axes[1].set_title(
    f"Contributions → hidden {HIDDEN_NODE}"
)

axes[1].axis(
    "off"
)

figure.colorbar(
    image,
    ax=axes[1],
    label="Edge contribution"
)

plt.tight_layout()

plt.show()