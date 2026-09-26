import numpy as np

from kumo.network import KumoNetwork

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

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

predictions = network.forward(X)

print("Input shape:")
print(X.shape)

print("\nPrediction shape:")
print(predictions.shape)

print("\nPredictions:")
print(predictions)

error = predictions - Y
gradients = network.backward(
    error
)

print("\nError shape:")
print(error.shape)

print("\nGradient shapes:")

for index, gradient in enumerate(
    gradients
):
    print(
        f"Layer {index + 1}: "
        f"{gradient.shape}"
    )

print("\nExpected:")
print("Predictions: (4, 1)")
print("Layer 1 gradient: (2, 3, 3)")
print("Layer 2 gradient: (3, 1, 3)")