import numpy as np

from kumo.layer import KumoLayer


np.random.seed(42)

layer = KumoLayer(
    n_inputs=2,
    n_outputs=1,
    degree=2
)

x = np.array([
    2.0,
    3.0
])

target = np.array([
    5.0
])

learning_rate = 0.001
epochs = 1000

for epoch in range(epochs):
    prediction = layer.forward(x)

    error = prediction - target

    loss = 0.5 * np.sum(
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
            f"Loss: {loss:.6f} | "
            f"Prediction: "
            f"{prediction[0]:.6f}"
        )

prediction = layer.forward(x)

print("\nFinal prediction:")
print(prediction)

print("\nTarget:")
print(target)

print("\nLearned coefficients:")
print(layer.C)

print("\nLearned bias:")
print(layer.b)