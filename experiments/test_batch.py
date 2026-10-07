import numpy as np

from kumo.network import KumoNetwork


np.random.seed(42)

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

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

predictions = network.forward(X)

error = predictions - y

gradients = network.backward(error)

print("Input shape:")
print(X.shape)

print("\nPrediction shape:")
print(predictions.shape)

print("\nPredictions:")
print(predictions)

print("\nError shape:")
print(error.shape)

print("\nGradient shapes:")

for i, (
    coefficient_gradients,
    bias_gradients
) in enumerate(gradients):

    print(
        f"Layer {i + 1} "
        f"coefficients: "
        f"{coefficient_gradients.shape}"
    )

    print(
        f"Layer {i + 1} "
        f"bias: "
        f"{bias_gradients.shape}"
    )

network.update(
    gradients,
    learning_rate=0.001
)

new_predictions = network.forward(X)

print("\nPredictions after one update:")
print(new_predictions)

print("\nMaximum prediction change:")
print(
    np.max(
        np.abs(
            new_predictions
            - predictions
        )
    )
)