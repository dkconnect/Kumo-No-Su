import numpy as np
from kumo.network import KumoNetwork

# reporducibility 
np.random.seed(42)
EPSILON = 1e-5

# small test
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

# loss fn
def calculate_loss():
    prediction = network.forward(x)
    error = prediction - target
    loss = 0.5 * np.sum(
        error ** 2
    )
    return loss

# analytical gradient 
prediction = network.forward(x)
error = prediction - target

analytical_gradients = network.backward(
    error
)

# C[I][O][COEFF]
layer_index = 0
input_index = 0
output_index = 0
coefficient_index = 2

layer = network.layers[
    layer_index
]

original_value = layer.C[
    input_index,
    output_index,
    coefficient_index
]


# c+
layer.C[
    input_index,
    output_index,
    coefficient_index
] = (
    original_value + EPSILON
)
loss_plus = calculate_loss()


# c MINUS

layer.C[
    input_index,
    output_index,
    coefficient_index
] = (
    original_value - EPSILON
)
loss_minus = calculate_loss()

layer.C[
    input_index,
    output_index,
    coefficient_index
] = original_value

# num gradient
numerical_gradient = (
    loss_plus - loss_minus
) / (
    2.0 * EPSILON
)

analytical_gradient = (
    analytical_gradients[
        layer_index
    ][
        input_index,
        output_index,
        coefficient_index
    ]
)

difference = abs(
    analytical_gradient
    - numerical_gradient
)

print("\nKumo Gradient Check")

print(
    "Coefficient:",
    f"Layer {layer_index + 1}, "
    f"C[{input_index}, "
    f"{output_index}, "
    f"{coefficient_index}]"
)

print()

print(
    f"Analytical gradient: "
    f"{analytical_gradient:.10f}"
)

print(
    f"Numerical gradient:  "
    f"{numerical_gradient:.10f}"
)

print(
    f"Difference:          "
    f"{difference:.10e}"
)

print()

if difference < 1e-6:
    print(
        "PASS: Kumo backprop agrees "
        "with the numerical derivative."
    )

else:
    print(
        "FAIL: The gradients do not "
        "agree closely enough."
    )