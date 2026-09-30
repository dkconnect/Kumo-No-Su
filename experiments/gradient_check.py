import numpy as np
from kumo.network import KumoNetwork

np.random.seed(42)

EPSILON = 1e-5
TOLERANCE = 1e-6

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
    0.7
])

target = np.array([
    0.8
])


def calculate_loss():
    prediction = network.forward(x)
    error = prediction - target
    return 0.5 * np.sum(
        error ** 2
    )


prediction = network.forward(x)
error = prediction - target
analytical_gradients = network.backward(
    error
)

print()
print("Kumo Full Gradient Check")
print()

total_checked = 0
failed = 0

largest_difference = 0.0
largest_location = None

for layer_index, layer in enumerate(
    network.layers
):
    print(
        f"Checking Layer "
        f"{layer_index + 1}..."
    )

    layer_gradient = (
        analytical_gradients[
            layer_index
        ]
    )

    for input_index in range(
        layer.n_inputs
    ):

        for output_index in range(
            layer.n_outputs
        ):

            for coefficient_index in range(
                layer.n_coeffs
            ):

                original_value = layer.C[
                    input_index,
                    output_index,
                    coefficient_index
                ]

                layer.C[
                    input_index,
                    output_index,
                    coefficient_index
                ] = (
                    original_value
                    + EPSILON
                )
                loss_plus = calculate_loss()

                layer.C[
                    input_index,
                    output_index,
                    coefficient_index
                ] = (
                    original_value
                    - EPSILON
                )
                loss_minus = calculate_loss()

                layer.C[
                    input_index,
                    output_index,
                    coefficient_index
                ] = original_value

                numerical_gradient = (
                    loss_plus
                    - loss_minus
                ) / (
                    2.0 * EPSILON
                )

                analytical_gradient = (
                    layer_gradient[
                        input_index,
                        output_index,
                        coefficient_index
                    ]
                )

                difference = abs(
                    analytical_gradient
                    - numerical_gradient
                )

                total_checked += 1
                if difference > largest_difference:

                    largest_difference = (
                        difference
                    )

                    largest_location = (
                        layer_index,
                        input_index,
                        output_index,
                        coefficient_index
                    )

                if difference > TOLERANCE:

                    failed += 1
                    print(
                        "FAIL"
                        f"C[{input_index}, "
                        f"{output_index}, "
                        f"{coefficient_index}]"
                    )

                    print("analytical = "f"{analytical_gradient:.10f}")
                    print("numerical  = "f"{numerical_gradient:.10f}")
                    print("difference = "f"{difference:.10e}")

    print(f"Layer {layer_index + 1} "f"finished.")
    print()

print()
print("Gradient Check Summary")
print()

print(f"Coefficients checked: "f"{total_checked}")
print(f"Failed: "f"{failed}")
print(f"Largest difference: "f"{largest_difference:.10e}"
)

if largest_location is not None:

    (
        layer_index,
        input_index,
        output_index,
        coefficient_index
    ) = largest_location

    print(
        "Largest difference at: "
        f"Layer {layer_index + 1}, "
        f"C[{input_index}, "
        f"{output_index}, "
        f"{coefficient_index}]"
    )
print()

if failed == 0:
    print("PASS: All Kumo gradients agree with numerical derivatives.")

else:
    print("FAIL: Some Kumo gradients do not match numerical derivatives.")