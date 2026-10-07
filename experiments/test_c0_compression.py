import numpy as np

from data.load_mnist import load_mnist_split


MODEL_PATH = "models/mnist_full_kumo.npz"

X_train, y_train, X_test, y_test = (
    load_mnist_split()
)

data = np.load(
    MODEL_PATH
)

n_layers = int(
    data["n_layers"][0]
)

if "format_version" in data.files:
    raise ValueError(
        "test_c0_compression.py expects "
        "a historical v1 model"
    )

old_layers = []

for layer_index in range(
    n_layers
):
    old_C = data[
        f"layer_{layer_index}_C"
    ]

    old_layers.append(
        old_C
    )


def v1_forward(coefficients, x):
    output = np.zeros(
        coefficients.shape[1],
        dtype=float
    )

    for degree in range(
        coefficients.shape[2]
    ):
        output += np.sum(
            coefficients[:, :, degree]
            * x[:, None] ** degree,
            axis=0
        )

    return output


def compressed_forward(coefficients, x):
    bias = np.sum(
        coefficients[:, :, 0],
        axis=0
    )

    output = bias.copy()

    for degree in range(
        1,
        coefficients.shape[2]
    ):
        output += np.sum(
            coefficients[:, :, degree]
            * x[:, None] ** degree,
            axis=0
        )

    return output


maximum_hidden_difference = 0.0
maximum_output_difference = 0.0
prediction_mismatches = 0
normal_correct = 0
compressed_correct = 0

layer1_C = old_layers[0]
layer2_C = old_layers[1]

for x, target in zip(
    X_test,
    y_test
):
    normal_hidden = v1_forward(
        layer1_C,
        x
    )

    normal_output = v1_forward(
        layer2_C,
        normal_hidden
    )

    compressed_hidden = compressed_forward(
        layer1_C,
        x
    )

    compressed_output = compressed_forward(
        layer2_C,
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