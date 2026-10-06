import numpy as np

from kumo.layer import KumoLayer


np.random.seed(42)

layer = KumoLayer(
    n_inputs=2,
    n_outputs=3,
    degree=2
)

x = np.array([
    0.5,
    -0.25
])

output = layer.forward(
    x
)

error = np.array([
    1.0,
    -0.5,
    0.25
])

(
    coefficient_gradients,
    bias_gradients,
    input_gradients
) = layer.backward(
    error
)

print(
    "Coefficient shape:",
    layer.C.shape
)

print(
    "Bias shape:",
    layer.b.shape
)

print(
    "\nOutput shape:",
    output.shape
)

print(
    "Coefficient gradient shape:",
    coefficient_gradients.shape
)

print(
    "Bias gradient shape:",
    bias_gradients.shape
)

print(
    "Input gradient shape:",
    input_gradients.shape
)

print(
    "\nBias gradients:"
)

print(
    bias_gradients
)

print(
    "\nExpected bias gradients:"
)

print(
    error
)