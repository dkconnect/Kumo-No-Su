import os
import numpy as np

from data.load_mnist import load_mnist
from kumo.network import KumoNetwork
from kumo.losses import (
    softmax,
    cross_entropy,
    softmax_cross_entropy_gradient
)

np.random.seed(42)

TRAIN_SIZE = 2000
TEST_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


# load MNIST

print("Loading MNIST")

X, y = load_mnist()
X_train = X[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]

X_test = X[TRAIN_SIZE: TRAIN_SIZE + TEST_SIZE]
y_test = y[TRAIN_SIZE: TRAIN_SIZE + TEST_SIZE]

print()
print("Training images:")
print(X_train.shape)

print("Training labels:")
print(y_train.shape)

print()

print("Test images:")
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

print()

print("Layer 1 coefficients:", network.layers[0].C.shape)

print("Layer 2 coefficients:", network.layers[1].C.shape)

# calculate accuracy

def calculate_accuracy(
    network,
    X,
    y
):

    logits = network.forward(
        X
    )

    probabilities = softmax(
        logits
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return np.mean(
        predictions == y
    )

initial_accuracy = (
    calculate_accuracy(
        network,
        X_test,
        y_test
    )
)

print()
print("Initial test accuracy: "f"{initial_accuracy * 100:.2f}%")

n_samples = len(
    X_train
)

for epoch in range(EPOCHS):
    indices = np.random.permutation(n_samples)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0.0
    batches = 0

    for start in range(
        0,
        n_samples,
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE
        X_batch = X_train[
            start:end
        ]

        Y_batch = Y_train[
            start:end
        ]

        logits = network.forward(
            X_batch
        )

        probabilities = softmax(
            logits
        )

        loss = cross_entropy(
            probabilities, Y_batch
        )

        epoch_loss += loss
        batches += 1

        output_gradient = (
            softmax_cross_entropy_gradient(
                probabilities,
                Y_batch
            )
        )

        gradients = network.backward(
            output_gradient
        )

        network.update(
            gradients, LEARNING_RATE
        )

    average_loss = (
        epoch_loss
        / batches
    )

    test_accuracy = (
        calculate_accuracy(
            network,
            X_test,
            y_test
        )
    )

    print(
        f"Epoch {epoch + 1:2d} "
        f"| Loss: {average_loss:.6f} "
        f"| Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

print()
print("Sample predictions:")

sample_X = X_test[:10]
sample_logits = network.forward(sample_X)
sample_probabilities = softmax(sample_logits)
sample_predictions = np.argmax(sample_probabilities, axis=1)

for true_label, predicted_label, probs in zip(
    y_test[:10],
    sample_predictions,
    sample_probabilities
):
    confidence = probs[predicted_label]
    print(
        f"True: {true_label} "
        f"| Predicted: {predicted_label} "
        f"| Confidence: "
        f"{confidence * 100:.2f}%"
    )

os.makedirs("models",exist_ok=True)
model_path = (
    "models/" 
    "mnist_softmax_kumo.npz"
    )

network.save(model_path)

print()
print("Saved model to:",model_path)