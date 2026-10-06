import numpy as np
import matplotlib.pyplot as plt

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

powers = np.stack(
    [
        x ** degree
        for degree in range(
            layer.n_coeffs
        )
    ],
    axis=1
)

contributions = np.sum(
    coefficients * powers,
    axis=1
)

print("\nHidden node:", HIDDEN_NODE)

print(
    "Hidden activation:",
    hidden[HIDDEN_NODE]
)

print(
    "Sum of edge contributions:",
    np.sum(contributions)
)

print(
    "Difference:",
    abs(
        hidden[HIDDEN_NODE]
        - np.sum(contributions)
    )
)

contribution_image = contributions.reshape(
    28,
    28
)

limit = np.max(
    np.abs(contribution_image)
)

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    contribution_image,
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(
    label="Edge Contribution"
)

plt.title(
    f"Contributions to Hidden Node "
    f"{HIDDEN_NODE}"
)

plt.axis("off")

plt.tight_layout()
plt.show()