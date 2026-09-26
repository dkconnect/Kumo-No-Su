import numpy as np
from kumo.network import KumoNetwork

# network create

network = KumoNetwork()

network.add_layer(
    n_inputs=2,
    n_outputs=3,
    degree=1
)

network.add_layer(
    n_inputs=3,
    n_outputs=1,
    degree=1
)


# data for training

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


# training settings 
learning_rate = 0.001
epochs = 1000


# train 

for epoch in range(epochs):
    total_loss = 0.0
    for x, target in zip(X, Y):

        # Forward
        prediction = network.forward(x)

        # Loss
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

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Total Loss: {total_loss:.6f}"
        )


# training prediction

print("\nTraining predictions:")

for x, target in zip(X, Y):

    prediction = network.forward(x)

    print(
        f"{x} -> "
        f"prediction: {prediction[0]:.4f}, "
        f"target: {target[0]:.1f}"
    )

test_x = np.array([
    6.0,
    2.0
])

test_prediction = network.forward(
    test_x
)

print("\nUnseen example:")

print(
    f"{test_x} -> "
    f"prediction: {test_prediction[0]:.4f}"
)

print("Expected: 8.0")