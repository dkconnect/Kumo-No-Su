import matplotlib.pyplot as plt
import numpy as np

from data.load_mnist import load_mnist_split
from kumo.network import KumoNetwork


MODEL_PATH = "models/mnist_full_kumo.npz"
HIDDEN_NODE = 5
N_EDGES = 12

'''
Every line is a different learned function:
Φ_{ij}(x)

and every dot is where the current digit activates that particular strand.

A conventional dense layer would effectively give us straight lines:
Φ(x)=wx

Kumo degree 2 can give us upward curves, downward curves, nearly-linear strands, positive responses and negative responses.
'''

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

important_edges = np.argsort(
    np.abs(contributions)
)[-N_EDGES:]

important_edges = important_edges[
    ::-1
]

x_values = np.linspace(
    0.0,
    1.0,
    200
)


print(
    "\nHidden node:",
    HIDDEN_NODE
)

print(
    "Bias:",
    layer.b[HIDDEN_NODE]
)

print(
    "\nMost influential edges:"
)

print(
    "Pixel | Location | Value | Contribution"
)

print(
    "---------------------------------------"
)


for pixel_index in important_edges:

    row = pixel_index // 28
    column = pixel_index % 28

    print(
        f"{pixel_index:5d} | "
        f"({row:2d}, {column:2d}) | "
        f"{x[pixel_index]:5.3f} | "
        f"{contributions[pixel_index]: .6f}"
    )


for pixel_index in important_edges:

    edge_coefficients = coefficients[
        pixel_index
    ]

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

    plt.plot(
        x_values,
        curve,
        alpha=0.8,
        label=str(pixel_index)
    )

    actual_x = x[
        pixel_index
    ]

    actual_y = contributions[
        pixel_index
    ]

    plt.scatter(
        [actual_x],
        [actual_y],
        s=25
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
    f"Kumo No Su — strongest edges → hidden {HIDDEN_NODE}"
)

plt.legend(
    title="Pixel",
    bbox_to_anchor=(
        1.05,
        1
    ),
    loc="upper left"
)

plt.tight_layout()

plt.show()