import numpy as np
from kumo.layer import KumoLayer

layer = KumoLayer(
    n_inputs=2,
    n_outputs=1,
    degree=1
)

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

learning_rate = 0.001

for epoch in range(1000):
    total_loss = 0.0
    for x, target in zip(X, Y):
        prediction = layer.forward(x)
        error = prediction - target
        loss = 0.5 * np.sum(error ** 2)

        coefficient_gradients, bias_gradients = layer.backward(error)
        layer.update(
            coefficient_gradients,
            bias_gradients,
            learning_rate
        )
        total_loss += loss

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Total Loss: {total_loss:.6f}")

print("\nTraining predictions:")

for x, target in zip(X, Y):
    prediction = layer.forward(x)

    print(
        f"{x} -> "
        f"prediction: {prediction[0]:.4f}, "
        f"target: {target[0]:.1f}"
    )
test_x = np.array([6.0, 2.0])

test_prediction = layer.forward(test_x)

print("\nUnseen example:")
print(f"{test_x} -> prediction: {test_prediction[0]:.4f}")
print("Expected: 8.0") 

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("\nFirst input:", X[0])
print("First target:", Y[0])