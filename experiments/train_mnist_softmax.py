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

TRAIN_SIZE = 10000
VALIDATION_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "models/mnist_10k_kumo.npz"

X, y = load_mnist()

X_train = X[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]

X_validation = X[TRAIN_SIZE: TRAIN_SIZE + VALIDATION_SIZE]
y_validation = y[TRAIN_SIZE: TRAIN_SIZE + VALIDATION_SIZE]

print("\nTraining images:")
print(X_train.shape)

print("Training labels:")
print(y_train.shape)

print("\nValidation images:")
print(X_validation.shape)

print("Validation labels:")
print(y_validation.shape)


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


Y_train = one_hot(y_train)
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

print(
    "Layer 1 coefficients:",
    network.layers[0].C.shape
)

print(
    "Layer 2 coefficients:",
    network.layers[1].C.shape
)

def calculate_accuracy(
    network,
    X,
    labels
):
    logits = network.forward(X)
    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

initial_accuracy = calculate_accuracy(
    network,
    X_validation,
    y_validation
)

print(
    "\nInitial validation accuracy: "
    f"{initial_accuracy * 100:.2f}%"
)

os.makedirs("models", exist_ok=True)

best_accuracy = initial_accuracy
best_epoch = 0

network.save(MODEL_PATH)

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

        logits = network.forward(X_batch)
        probabilities = softmax(logits)
        batch_loss = cross_entropy(probabilities, Y_batch)

        total_loss += batch_loss
        n_batches += 1

        output_gradient = (
            softmax_cross_entropy_gradient(
                probabilities,
                Y_batch
            )
        )

        gradients = network.backward(output_gradient)
        network.update(gradients, LEARNING_RATE)

    average_loss = (total_loss / n_batches)

    validation_accuracy = (
        calculate_accuracy(
            network,
            X_validation,
            y_validation
        )
    )

    print(
        f"Epoch {epoch + 1:2d} | "
        f"Loss: {average_loss:.6f} | "
        f"Validation accuracy: "
        f"{validation_accuracy * 100:.2f}%",
        end=""
    )

    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_epoch = epoch + 1

        network.save(MODEL_PATH)

        print("SAVED")
    else:
        print()


print("\nTraining complete.")

print(f"Best epoch: {best_epoch}")

print(
    "Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Best model saved to: "
    f"{MODEL_PATH}"
)

best_network = KumoNetwork.load(MODEL_PATH)

print(
    "\nSample predictions "
    "from best checkpoint:"
)

sample_logits = best_network.forward(X_validation[:10])
sample_probabilities = softmax(sample_logits)
sample_predictions = np.argmax(sample_probabilities, axis=1)

for i in range(10):
    predicted = sample_predictions[i]
    confidence = (
        sample_probabilities[
            i,
            predicted
        ]
    )

    print(
        f"True: {y_validation[i]} | "
        f"Predicted: {predicted} | "
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )