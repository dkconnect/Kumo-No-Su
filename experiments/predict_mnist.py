import numpy as np
import matplotlib.pyplot as plt

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork
from kumo.losses import softmax

MODEL_PATH = "models/mnist_full_kumo.npz"
X_train, y_train, X_test, y_test = (load_mnist_split())
network = KumoNetwork.load(MODEL_PATH)

print("\nLoaded Kumo model:")
print(MODEL_PATH)
print("\nArchitecture:")

for i, layer in enumerate(network.layers):
    print(
        f"Layer {i + 1}: "
        f"{layer.n_inputs} -> "
        f"{layer.n_outputs} "
        f"(degree {layer.degree})"
    )


index = np.random.randint(0, len(X_test))
image = X_test[index]
true_label = y_test[index]

logits = network.forward(image)
probabilities = softmax(logits)
prediction = np.argmax(probabilities)

print("\nKumo Prediction")
print(f"Test image index: {index}")
print(f"Actual digit: {true_label}")
print(f"Predicted digit: {prediction}")
print(f"Confidence: "f"{probabilities[prediction] * 100:.2f}%")
print("\nProbabilities:")

for digit in range(10):
    marker = ""
    if digit == prediction:
        marker = "  <---"
    print(
        f"{digit}: "
        f"{probabilities[digit] * 100:6.2f}%"
        f"{marker}"
    )

image_2d = image.reshape(28, 28)
plt.imshow(image_2d, cmap="gray")

plt.title(
    f"Actual: {true_label} | "
    f"Kumo: {prediction} | "
    f"{probabilities[prediction] * 100:.1f}%"
)

plt.axis("off")
plt.show()