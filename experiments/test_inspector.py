import numpy as np

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork
from kumo.losses import softmax


MODEL_PATH = "models/mnist_full_kumo.npz"

X_train, y_train, X_test, y_test = (load_mnist_split())
network = KumoNetwork.load(MODEL_PATH)
x = X_test[0]

activations = network.inspect(x)

inputs = activations[0]
hidden = activations[1]
logits = activations[2]

probabilities = softmax(logits)
prediction = np.argmax(probabilities)

print()
print("Kumo Inspector")

print("Input:")
print("Shape:", inputs.shape)

print("Hidden layer:")
print("Shape:", hidden.shape)
print(hidden)

print("Output logits:")
print("Shape:", logits.shape)
print(logits)

print("Probabilities:")

for digit in range(10):
    print(
        f"{digit}: "
        f"{probabilities[digit] * 100:.2f}%"
    )

print()
print("Actual: ",y_test[0])

print("Predicted: ",prediction)