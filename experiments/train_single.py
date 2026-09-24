import numpy as np
from kumo.layer import KumoLayer

layer = KumoLayer(
    n_inputs=2,
    n_outputs=1,
    degree=2
)

x = np.array([2.0, 3.0])
target = np.array([5.0])

learning_rate = 0.001

for epoch in range(100):
    # Forward
    prediction = layer.forward(x)

    # Error and loss
    error = prediction - target
    loss = 0.5 * np.sum(error ** 2)

    # Backward
    coefficient_gradients, bias_gradients = layer.backward(error)

    layer.update(
        coefficient_gradients,
        bias_gradients,
        learning_rate
    )

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Prediction: {prediction[0]:.6f} | "
            f"Loss: {loss:.6f}"
        )