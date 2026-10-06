import matplotlib.pyplot as plt
import numpy as np

from data.load_mnist import load_mnist_split
from kumo.losses import softmax
from kumo.network import KumoNetwork


MODEL_PATH = "models/mnist_full_kumo.npz"
N_PIXELS = 10

X_train, y_train, X_test, y_test = (
    load_mnist_split()
)

network = KumoNetwork.load(
    MODEL_PATH
)

x = X_test[0]

activations = network.inspect(
    x
)

hidden = activations[1]
logits = activations[2]

probabilities = softmax(
    logits
)

prediction = np.argmax(
    probabilities
)


output_layer = network.layers[1]

output_coefficients = output_layer.C[
    :,
    prediction,
    :
]

output_degrees = np.arange(
    1,
    output_layer.degree + 1
)

hidden_powers = (
    hidden[:, None]
    ** output_degrees
)

hidden_contributions = np.sum(
    output_coefficients
    * hidden_powers,
    axis=1
)

important_hidden = np.argmax(
    np.abs(
        hidden_contributions
    )
)


hidden_layer = network.layers[0]

pixel_coefficients = hidden_layer.C[
    :,
    important_hidden,
    :
]

pixel_degrees = np.arange(
    1,
    hidden_layer.degree + 1
)

pixel_powers = (
    x[:, None]
    ** pixel_degrees
)

pixel_contributions = np.sum(
    pixel_coefficients
    * pixel_powers,
    axis=1
)

important_pixels = np.argsort(
    np.abs(
        pixel_contributions
    )
)[-N_PIXELS:]

important_pixels = important_pixels[
    ::-1
]


hidden_reconstructed = (
    hidden_layer.b[
        important_hidden
    ]
    + np.sum(
        pixel_contributions
    )
)

output_reconstructed = (
    output_layer.b[
        prediction
    ]
    + np.sum(
        hidden_contributions
    )
)


print(
    "\nActual digit:",
    y_test[0]
)

print(
    "Predicted digit:",
    prediction
)

print(
    "Confidence:",
    f"{probabilities[prediction] * 100:.2f}%"
)

print(
    "\nStrongest hidden path:"
)

print(
    "Hidden node:",
    important_hidden
)

print(
    "Hidden activation:",
    hidden[important_hidden]
)

print(
    "Contribution to prediction:",
    hidden_contributions[
        important_hidden
    ]
)

print(
    "\nHidden reconstruction:"
)

print(
    "Bias:",
    hidden_layer.b[
        important_hidden
    ]
)

print(
    "Edge contribution sum:",
    np.sum(
        pixel_contributions
    )
)

print(
    "Reconstructed:",
    hidden_reconstructed
)

print(
    "Difference:",
    abs(
        hidden[important_hidden]
        - hidden_reconstructed
    )
)

print(
    "\nOutput reconstruction:"
)

print(
    "Bias:",
    output_layer.b[
        prediction
    ]
)

print(
    "Hidden contribution sum:",
    np.sum(
        hidden_contributions
    )
)

print(
    "Reconstructed:",
    output_reconstructed
)

print(
    "Difference:",
    abs(
        logits[prediction]
        - output_reconstructed
    )
)


print(
    "\nStrongest pixel contributions:"
)

print(
    "Pixel | Location | Value | Contribution"
)

print(
    "---------------------------------------"
)

for pixel in important_pixels:

    row = pixel // 28
    column = pixel % 28

    print(
        f"{pixel:5d} | "
        f"({row:2d}, {column:2d}) | "
        f"{x[pixel]:5.3f} | "
        f"{pixel_contributions[pixel]: .6f}"
    )


contribution_image = (
    pixel_contributions.reshape(
        28,
        28
    )
)

limit = np.max(
    np.abs(
        contribution_image
    )
)

figure, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)


axes[0].imshow(
    x.reshape(28, 28),
    cmap="gray"
)

axes[0].set_title(
    f"Input: {y_test[0]}"
)

axes[0].axis(
    "off"
)


image = axes[1].imshow(
    contribution_image,
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit
)

axes[1].set_title(
    f"Pixels → hidden {important_hidden}"
)

axes[1].axis(
    "off"
)


axes[2].bar(
    np.arange(
        len(hidden)
    ),
    hidden_contributions
)

axes[2].axhline(
    0,
    linewidth=1
)

axes[2].set_xlabel(
    "Hidden node"
)

axes[2].set_ylabel(
    "Contribution"
)

axes[2].set_title(
    f"Hidden → digit {prediction}"
)


figure.colorbar(
    image,
    ax=axes[1],
    label="Pixel contribution"
)

plt.tight_layout()

plt.show()