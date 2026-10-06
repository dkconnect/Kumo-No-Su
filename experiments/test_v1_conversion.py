import numpy as np

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork


MODEL_PATH = "models/mnist_full_kumo.npz"

X_train, y_train, X_test, y_test = (
    load_mnist_split()
)

network = KumoNetwork.load(
    MODEL_PATH
)

print(
    "\nConverted architecture:"
)

for i, layer in enumerate(
    network.layers
):

    print(
        f"Layer {i + 1}"
    )

    print(
        "C:",
        layer.C.shape
    )

    print(
        "b:",
        layer.b.shape
    )


correct = 0

for x, target in zip(
    X_test,
    y_test
):

    logits = network.forward(
        x
    )

    prediction = np.argmax(
        logits
    )

    if prediction == target:
        correct += 1


accuracy = (
    correct
    / len(X_test)
)


print(
    "\nImages tested:",
    len(X_test)
)

print(
    "Accuracy:",
    f"{accuracy * 100:.2f}%"
)