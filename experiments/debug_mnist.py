import numpy as np

from data.load_mnist import load_mnist
from kumo.network import KumoNetwork


np.random.seed(42)

BATCH_SIZE = 32
LEARNING_RATE = 0.001

X, y = load_mnist()

X_train = X[:50000]
y_train = y[:50000]

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


def one_hot(labels):
    targets = np.zeros(
        (len(labels), 10)
    )

    targets[
        np.arange(len(labels)),
        labels
    ] = 1.0

    return targets


def stats(name, array):
    print(
        f"{name:<18} "
        f"min={np.min(array): .5e}  "
        f"max={np.max(array): .5e}  "
        f"mean={np.mean(array): .5e}  "
        f"finite={np.all(np.isfinite(array))}"
    )


print("\nDebugging first epoch...\n")

for batch_start in range(
    0,
    len(X_train),
    BATCH_SIZE
):
    batch_end = (
        batch_start
        + BATCH_SIZE
    )

    X_batch = X_train[
        batch_start:batch_end
    ]

    y_batch = y_train[
        batch_start:batch_end
    ]

    targets = one_hot(
        y_batch
    )

    hidden = network.layers[
        0
    ].forward(X_batch)

    output = network.layers[
        1
    ].forward(hidden)

    error = output - targets

    gradients = network.backward(
        error
    )

    if batch_start == 0:
        print("--- Batch 0 ---")

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

        for i, (
            coefficient_gradients,
            bias_gradients
        ) in enumerate(gradients):

            stats(
                f"layer{i + 1} C grad",
                coefficient_gradients
            )

            stats(
                f"layer{i + 1} b grad",
                bias_gradients
            )

        stats(
            "layer1 C",
            network.layers[0].C
        )

        stats(
            "layer1 b",
            network.layers[0].b
        )

        stats(
            "layer2 C",
            network.layers[1].C
        )

        stats(
            "layer2 b",
            network.layers[1].b
        )

        print()

    network.update(
        gradients,
        LEARNING_RATE
    )

    if batch_start == 0:
        print(
            "After first update:"
        )

        stats(
            "layer1 C",
            network.layers[0].C
        )

        stats(
            "layer1 b",
            network.layers[0].b
        )

        stats(
            "layer2 C",
            network.layers[1].C
        )

        stats(
            "layer2 b",
            network.layers[1].b
        )

        break