import matplotlib.pyplot as plt
import numpy as np

from data.load_mnist import load_mnist_split
from kumo.losses import softmax
from kumo.network import KumoNetwork


MODEL_PATH = "models/mnist_full_kumo.npz"

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
logits = activations[2]

probabilities = softmax(
    logits
)

prediction = np.argmax(
    probabilities
)

layer = network.layers[1]

coefficients = layer.C[
    :,
    prediction,
    :
]

degrees = np.arange(
    1,
    layer.degree + 1
)

powers = (
    hidden[:, None] ** degrees
)

contributions = np.sum(
    coefficients * powers,
    axis=1
)

bias = layer.b[
    prediction
]

reconstructed = (
    bias
    + np.sum(contributions)
)


print(
    "\nActual digit:",
    y_test[0]
)

print(
    "Predicted digit:",
    prediction
)

print(
    "Confidence:",
    f"{probabilities[prediction] * 100:.2f}%"
)

print(
    "\nOutput bias:",
    bias
)

print(
    "Hidden contribution sum:",
    np.sum(contributions)
)

print(
    "Actual logit:",
    logits[prediction]
)

print(
    "Reconstructed logit:",
    reconstructed
)

print(
    "Difference:",
    abs(
        logits[prediction]
        - reconstructed
    )
)


order = np.argsort(
    np.abs(contributions)
)[::-1]

print(
    "\nHidden node contributions:"
)

print(
    "Node | Activation | Contribution"
)

print(
    "--------------------------------"
)

for node in order:

    print(
        f"{node:4d} | "
        f"{hidden[node]:10.4f} | "
        f"{contributions[node]: .6f}"
    )


plt.bar(
    np.arange(
        len(hidden)
    ),
    contributions
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Hidden node"
)

plt.ylabel(
    "Contribution to output logit"
)

plt.title(
    f"Hidden nodes → digit {prediction}"
)

plt.xticks(
    np.arange(
        len(hidden)
    )
)

plt.tight_layout()

plt.show()