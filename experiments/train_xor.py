import numpy as np
from kumo.network import KumoNetwork

# XOR Data
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


# creating KUMO

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

# Training settings

learning_rate = 0.01
epochs = 5000

for epoch in range(epochs):
    total_loss = 0.0

    for x, target in zip(X, Y):
        prediction = network.forward(x)
        error = prediction - target
        loss = 0.5 * np.sum(
            error ** 2
        )

        total_loss += loss
        gradients = network.backward(error)

        network.update(
            gradients,
            learning_rate
        )

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {total_loss:.6f}"
        )

print("\nXOR predictions:")

for x, target in zip(X, Y):
    prediction = network.forward(x)

    print(
        f"{x} -> "
        f"{prediction[0]:.4f} "
        f"(target {target[0]:.0f})"
    )