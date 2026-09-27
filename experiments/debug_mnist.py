import numpy as np
from data.load_mnist import load_mnist
from kumo.network import KumoNetwork

TRAIN_SIZE = 2000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

np.seterr(
    over="ignore",
    invalid="ignore"
)

X, y = load_mnist()
X_train = X[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]

def one_hot(labels, n_classes=10):
    encoded = np.zeros(
        (len(labels), n_classes)
    )

    encoded[
        np.arange(len(labels)),
        labels
    ] = 1.0
    return encoded

Y_train = one_hot(y_train)
network = KumoNetwork()

network.add_layer(
    n_inputs=784,
    n_outputs=16,
    degree=2
)

network.add_layer(
    n_inputs=16,
    n_outputs=10,
    degree=2
)

def stats(name, array):
    finite = np.all(
        np.isfinite(array)
    )

    print(
        f"{name:<18} "
        f"min={np.nanmin(array):>12.5e}  "
        f"max={np.nanmax(array):>12.5e}  "
        f"mean={np.nanmean(array):>12.5e}  "
        f"finite={finite}"
    )

print("\nDebugging first epoch...\n")

for batch_number, start in enumerate(
    range(0, TRAIN_SIZE, BATCH_SIZE)
):

    end = start + BATCH_SIZE
    X_batch = X_train[start:end]
    Y_batch = Y_train[start:end]

    hidden = network.layers[0].forward(
        X_batch
    )
    output = network.layers[1].forward(
        hidden
    )
    error = output - Y_batch

    gradients = network.backward(
        error
    )

    print(
        f"\n--- Batch {batch_number} ---"
    )

    stats(
        "input",
        X_batch
    )

    stats(
        "hidden",
        hidden
    )

    stats(
        "output",
        output
    )

    stats(
        "error",
        error
    )

    stats(
        "layer1 grad",
        gradients[0]
    )

    stats(
        "layer2 grad",
        gradients[1]
    )

    stats(
        "layer1 C",
        network.layers[0].C
    )

    stats(
        "layer2 C",
        network.layers[1].C
    )

    arrays = [hidden, output, error, gradients[0], gradients[1]]

    if not all(
        np.all(np.isfinite(array))
        for array in arrays
    ):
        print(
            "\n!!! NON-FINITE VALUE FOUND !!!"
        )

        print(
            f"Explosion occurred in "
            f"batch {batch_number}."
        )
        break

    network.update(
        gradients,
        LEARNING_RATE
    )