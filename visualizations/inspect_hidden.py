import numpy as np
import matplotlib.pyplot as plt

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork
from kumo.losses import softmax


MODEL_PATH = "models/mnist_full_kumo.npz"

X_train, y_train, X_test, y_test = (
    load_mnist_split()
)

network = KumoNetwork.load(
    MODEL_PATH
)

index = 0
x = X_test[index]

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

nodes = np.arange(
    len(hidden)
)

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    nodes,
    hidden
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    nodes
)

plt.xlabel(
    "Hidden Node"
)

plt.ylabel(
    "Activation"
)

plt.title(
    f"Kumo Hidden Activations | "
    f"Actual: {y_test[index]} | "
    f"Predicted: {prediction}"
)

plt.tight_layout()
plt.show()