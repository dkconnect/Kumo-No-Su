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


def compressed_forward(layer, x):

    coefficients = layer.C

    bias = np.sum(
        coefficients[:, :, 0],
        axis=0
    )

    output = bias.copy()

    for degree in range(
        1,
        layer.n_coeffs
    ):

        output += np.sum(
            coefficients[
                :,
                :,
                degree
            ]
            * x[:, None] ** degree,
            axis=0
        )

    return output


maximum_hidden_difference = 0.0
maximum_output_difference = 0.0
prediction_mismatches = 0
normal_correct = 0
compressed_correct = 0

for x, target in zip(
    X_test,
    y_test
):

    normal_activations = network.inspect(
        x
    )

    normal_hidden = normal_activations[1]
    normal_output = normal_activations[2]

    compressed_hidden = compressed_forward(
        network.layers[0],
        x
    )

    compressed_output = compressed_forward(
        network.layers[1],
        compressed_hidden
    )

    hidden_difference = np.max(
        np.abs(
            normal_hidden
            - compressed_hidden
        )
    )

    output_difference = np.max(
        np.abs(
            normal_output
            - compressed_output
        )
    )

    maximum_hidden_difference = max(
        maximum_hidden_difference,
        hidden_difference
    )

    maximum_output_difference = max(
        maximum_output_difference,
        output_difference
    )

    normal_prediction = np.argmax(
        normal_output
    )

    compressed_prediction = np.argmax(
        compressed_output
    )

    if normal_prediction != compressed_prediction:
        prediction_mismatches += 1

    if normal_prediction == target:
        normal_correct += 1

    if compressed_prediction == target:
        compressed_correct += 1


normal_accuracy = (
    normal_correct
    / len(X_test)
)

compressed_accuracy = (
    compressed_correct
    / len(X_test)
)

print(
    "\nImages tested:",
    len(X_test)
)

print(
    "\nMaximum hidden difference:",
    maximum_hidden_difference
)

print(
    "Maximum output difference:",
    maximum_output_difference
)

print(
    "\nPrediction mismatches:",
    prediction_mismatches
)

print(
    "\nNormal accuracy:",
    f"{normal_accuracy * 100:.2f}%"
)

print(
    "Compressed accuracy:",
    f"{compressed_accuracy * 100:.2f}%"
)