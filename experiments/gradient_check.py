import numpy as np

from kumo.network import KumoNetwork


np.random.seed(42)

epsilon = 1e-6
tolerance = 1e-5

network = KumoNetwork()

network.add_layer(
    n_inputs=2,
    n_outputs=3,
    degree=2
)

network.add_layer(
    n_inputs=3,
    n_outputs=1,
    degree=2
)

x = np.array([
    0.4,
    -0.7
])

target = np.array([
    0.8
])


def loss():
    prediction = network.forward(x)

    return 0.5 * np.sum(
        (prediction - target) ** 2
    )


prediction = network.forward(x)

error = prediction - target

analytical_gradients = network.backward(
    error
)

checked = 0
failed = 0
largest_difference = 0.0

print("Kumo Full Gradient Check")

for layer_index, layer in enumerate(
    network.layers
):

    print(
        f"\nChecking Layer "
        f"{layer_index + 1}..."
    )

    (
        coefficient_gradients,
        bias_gradients
    ) = analytical_gradients[
        layer_index
    ]

    for index in np.ndindex(
        layer.C.shape
    ):

        original = layer.C[index]

        layer.C[index] = (
            original + epsilon
        )

        loss_plus = loss()

        layer.C[index] = (
            original - epsilon
        )

        loss_minus = loss()

        layer.C[index] = original

        numerical_gradient = (
            loss_plus - loss_minus
        ) / (
            2.0 * epsilon
        )

        analytical_gradient = (
            coefficient_gradients[
                index
            ]
        )

        difference = abs(
            numerical_gradient
            - analytical_gradient
        )

        largest_difference = max(
            largest_difference,
            difference
        )

        checked += 1

        if difference > tolerance:
            failed += 1

            print(
                "FAILED coefficient "
                f"{index}"
            )

            print(
                "Analytical:",
                analytical_gradient
            )

            print(
                "Numerical:",
                numerical_gradient
            )

            print(
                "Difference:",
                difference
            )

    for index in np.ndindex(
        layer.b.shape
    ):

        original = layer.b[index]

        layer.b[index] = (
            original + epsilon
        )

        loss_plus = loss()

        layer.b[index] = (
            original - epsilon
        )

        loss_minus = loss()

        layer.b[index] = original

        numerical_gradient = (
            loss_plus - loss_minus
        ) / (
            2.0 * epsilon
        )

        analytical_gradient = (
            bias_gradients[
                index
            ]
        )

        difference = abs(
            numerical_gradient
            - analytical_gradient
        )

        largest_difference = max(
            largest_difference,
            difference
        )

        checked += 1

        if difference > tolerance:
            failed += 1

            print(
                "FAILED bias "
                f"{index}"
            )

            print(
                "Analytical:",
                analytical_gradient
            )

            print(
                "Numerical:",
                numerical_gradient
            )

            print(
                "Difference:",
                difference
            )

print("\nResults")
print("Parameters checked:", checked)
print("Failed:", failed)
print(
    "Largest difference:",
    largest_difference
)

if failed == 0:
    print("\nPASS")
else:
    print("\nFAIL")