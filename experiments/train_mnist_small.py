import os
import numpy as np
from data.load_mnist import load_mnist
from kumo.network import KumoNetwork

np.random.seed(42)
TRAIN_SIZE = 2000
TEST_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

X, y = load_mnist()

X_train = X[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]

X_test = X[
    TRAIN_SIZE:
    TRAIN_SIZE + TEST_SIZE
]

y_test = y[
    TRAIN_SIZE:
    TRAIN_SIZE + TEST_SIZE
]

print("\nTraining images:")
print(X_train.shape)

print("Training labels:")
print(y_train.shape)

print("\nTest images:")
print(X_test.shape)

print("Test labels:")
print(y_test.shape)

def one_hot(labels, n_classes=10):
    encoded = np.zeros(
        (
            len(labels),
            n_classes
        )
    )

    encoded[
        np.arange(len(labels)),
        labels
    ] = 1.0

    return encoded

Y_train = one_hot(
    y_train
)

network = KumoNetwork()
network.add_layer(
    n_inputs=784,
    n_outputs=16,
    degree=2
)

network.add_layer(
    n_inputs=16,
    n_outputs=10,
    degree=2
)

print("\nKumo architecture:")
print("784 -> 16 -> 10")

print("Layer 1 coefficients:",network.layers[0].C.shape)
print("Layer 2 coefficients:",network.layers[1].C.shape)

def calculate_accuracy(
    network,
    X,
    labels
):
    outputs = network.forward(X)
    predictions = np.argmax(
        outputs,
        axis=1
    )

    accuracy = np.mean(
        predictions == labels
    )
    return accuracy

initial_accuracy = calculate_accuracy(
    network,
    X_test,
    y_test
)

print("\nInitial test accuracy: "f"{initial_accuracy * 100:.2f}%")

n_samples = len(X_train)
for epoch in range(EPOCHS):

    indices = np.random.permutation(
        n_samples
    )

    X_train = X_train[indices]
    Y_train = Y_train[indices]
    y_train = y_train[indices]

    total_loss = 0.0
    n_batches = 0

    for start in range(
        0,
        n_samples,
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE
        X_batch = X_train[start:end]

        Y_batch = Y_train[start:end]
        outputs = network.forward(X_batch)
        error = (outputs - Y_batch)

        batch_loss = np.mean(
            0.5
            * np.sum(
                error ** 2,
                axis=1
            )
        )

        total_loss += batch_loss
        n_batches += 1
        gradients = network.backward(
            error
        )

        network.update(gradients, LEARNING_RATE)

    average_loss = (total_loss / n_batches)

    test_accuracy = (
        calculate_accuracy(
            network,
            X_test,
            y_test
        )
    )

    print(
        f"Epoch {epoch + 1:2d} | "
        f"Loss: {average_loss:.6f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

print("\nSample predictions:")
sample_outputs = network.forward(X_test[:10])

sample_predictions = np.argmax(sample_outputs, axis=1)

for i in range(10):
    print(
        f"True: {y_test[i]} | "
        f"Predicted: "
        f"{sample_predictions[i]}"
    )

os.makedirs("models", exist_ok=True)
model_path = ("models/mnist_small_kumo.npz")
network.save(model_path)

print(f"\nSaved model to: f"{model_path}"
)