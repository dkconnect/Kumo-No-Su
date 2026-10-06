import numpy as np

from kumo.layer import KumoLayer


np.random.seed(42)

EPSILON = 1e-6
TOLERANCE = 1e-5

layer = KumoLayer(
    n_inputs=2,
    n_outputs=3,
    degree=2
)

x = np.array([
    0.4,
    -0.7
])

target = np.array([
    0.2,
    -0.3,
    0.5
])


def loss():
    output = layer.forward(x)

    return 0.5 * np.sum(
        (output - target) ** 2
    )


output = layer.forward(x)

error = (
    output - target
)

(
    coefficient_gradients,
    bias_gradients,
    input_gradients
) = layer.backward(
    error
)


largest_difference = 0.0
failed = 0
checked = 0


print("\nChecking coefficients")

for index in np.ndindex(
    layer.C.shape
):

    original = layer.C[index]

    layer.C[index] = (
        original + EPSILON
    )

    loss_plus = loss()

    layer.C[index] = (
        original - EPSILON
    )

    loss_minus = loss()

    layer.C[index] = original

    numerical = (
        loss_plus - loss_minus
    ) / (
        2 * EPSILON
    )

    analytical = (
        coefficient_gradients[index]
    )

    difference = abs(
        numerical - analytical
    )

    largest_difference = max(
        largest_difference,
        difference
    )

    checked += 1

    if difference > TOLERANCE:
        failed += 1

        print(
            "FAILED",
            index,
            numerical,
            analytical,
            difference
        )


print("\nChecking biases")

for index in np.ndindex(
    layer.b.shape
):

    original = layer.b[index]

    layer.b[index] = (
        original + EPSILON
    )

    loss_plus = loss()

    layer.b[index] = (
        original - EPSILON
    )

    loss_minus = loss()

    layer.b[index] = original

    numerical = (
        loss_plus - loss_minus
    ) / (
        2 * EPSILON
    )

    analytical = (
        bias_gradients[index]
    )

    difference = abs(
        numerical - analytical
    )

    largest_difference = max(
        largest_difference,
        difference
    )

    checked += 1

    if difference > TOLERANCE:
        failed += 1

        print(
            "FAILED",
            index,
            numerical,
            analytical,
            difference
        )


print("\nChecking inputs")

for index in np.ndindex(
    x.shape
):

    original = x[index]

    x[index] = (
        original + EPSILON
    )

    loss_plus = loss()

    x[index] = (
        original - EPSILON
    )

    loss_minus = loss()

    x[index] = original

    numerical = (
        loss_plus - loss_minus
    ) / (
        2 * EPSILON
    )

    analytical = (
        input_gradients[index]
    )

    difference = abs(
        numerical - analytical
    )

    largest_difference = max(
        largest_difference,
        difference
    )

    checked += 1

    if difference > TOLERANCE:
        failed += 1

        print(
            "FAILED",
            index,
            numerical,
            analytical,
            difference
        )


print(
    "\nValues checked:",
    checked
)

print(
    "Failed:",
    failed
)

print(
    "Largest difference:",
    largest_difference
)

if failed == 0:
    print(
        "\nPASS"
    )
else:
    print(
        "\nFAIL"
    )