import matplotlib.pyplot as plt
import numpy as np

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork


MODEL_PATH = "models/mnist_full_kumo.npz"
HIDDEN_NODE = 5

X_train, y_train, X_test, y_test = (
    load_mnist_split()
)

network = KumoNetwork.load(
    MODEL_PATH
)

x = X_test[0]

layer = network.layers[0]

coefficients = layer.C[
    :,
    HIDDEN_NODE,
    :
]

degrees = np.arange(
    1,
    layer.degree + 1
)

powers = (
    x[:, None] ** degrees
)

contributions = np.sum(
    coefficients * powers,
    axis=1
)

pixel_index = np.argmax(
    np.abs(contributions)
)

row = pixel_index // 28
column = pixel_index % 28

edge_coefficients = coefficients[
    pixel_index
]

actual_x = x[
    pixel_index
]

actual_contribution = contributions[
    pixel_index
]

x_values = np.linspace(
    0.0,
    1.0,
    200
)

curve = np.zeros_like(
    x_values
)

for degree, coefficient in enumerate(
    edge_coefficients,
    start=1
):

    curve += (
        coefficient
        * x_values ** degree
    )


print(
    "\nHidden node:",
    HIDDEN_NODE
)

print(
    "Pixel index:",
    pixel_index
)

print(
    "Pixel location:",
    (row, column)
)

print(
    "Pixel value:",
    actual_x
)

print(
    "\nEdge coefficients:"
)

for degree, coefficient in enumerate(
    edge_coefficients,
    start=1
):

    print(
        f"c{degree}:",
        coefficient
    )

print(
    "\nActual edge contribution:",
    actual_contribution
)


plt.plot(
    x_values,
    curve,
    label="Learned edge function"
)

plt.scatter(
    [actual_x],
    [actual_contribution],
    label="Current pixel"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Pixel intensity x"
)

plt.ylabel(
    "Edge output φ(x)"
)

plt.title(
    f"Kumo edge: pixel {pixel_index} → hidden {HIDDEN_NODE}"
)

plt.legend()

plt.show()