import numpy as np

from kumo.layer import KumoLayer


np.random.seed(42)

X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [4.0, 1.0],
    [3.0, 5.0]
])

Y = np.array([
    [3.0],
    [5.0],
    [5.0],
    [8.0]
])

layer = KumoLayer(
    n_inputs=2,
    n_outputs=1,
    degree=2
)

learning_rate = 0.001
epochs = 1000

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("\nFirst input:", X[0])
print("First target:", Y[0])

for epoch in range(epochs):
    predictions = layer.forward(X)

    error = predictions - Y

    loss = 0.5 * np.mean(
        error ** 2
    )

    (
        coefficient_gradients,
        bias_gradients,
        input_gradients
    ) = layer.backward(error)

    layer.update(
        coefficient_gradients,
        bias_gradients,
        learning_rate
    )

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss:.6f}"
        )

predictions = layer.forward(X)

print("\nTraining predictions:")

for x, prediction, target in zip(
    X,
    predictions,
    Y
):
    print(
        f"{x} -> "
        f"prediction: "
        f"{prediction[0]:.4f}, "
        f"target: {target[0]:.1f}"
    )

test_input = np.array([
    6.0,
    2.0
])

test_prediction = layer.forward(
    test_input
)

print("\nUnseen example:")
print(
    f"{test_input} -> "
    f"prediction: "
    f"{test_prediction[0]:.4f}"
)

print("Expected: 8.0")

print("\nLearned coefficients:")
print(layer.C)

print("\nLearned bias:")
print(layer.b)