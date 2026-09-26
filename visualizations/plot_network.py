import os
import numpy as np
import matplotlib.pyplot as plt
from kumo.network import KumoNetwork

MODEL_PATH = "models/xor_kumo.npz"
OUTPUT_PATH = "outputs/kumo_web.png"

# load trained kumo
network = KumoNetwork.load(
    MODEL_PATH
)

os.makedirs(
    "outputs",
    exist_ok=True
)

# node positions 

input_positions = [
    (0.0, 0.7),
    (0.0, 0.3)
]

hidden_positions = [
    (1.0, 0.8),
    (1.0, 0.5),
    (1.0, 0.2)
]

output_positions = [
    (2.0, 0.5)
]

# for formatting poly.

def format_polynomial(coefficients):
    terms = []

    for degree, coefficient in enumerate(
        coefficients
    ):
        if degree == 0:
            term = f"{coefficient:+.2f}"

        elif degree == 1:
            term = f"{coefficient:+.2f}x"

        else:
            term = (
                f"{coefficient:+.2f}"
                f"x^{degree}"
            )

        terms.append(term)

    return " ".join(terms)

def edge_strength(coefficients):
    """
    Simple visualization metric.
    """

    return np.linalg.norm(
        coefficients
    )

fig, ax = plt.subplots(
    figsize=(14, 8)
)

# drawing layer 1 edge

layer1 = network.layers[0]
for input_index in range(
    layer1.n_inputs
):

    for hidden_index in range(
        layer1.n_outputs
    ):
        start = input_positions[
            input_index
        ]
        end = hidden_positions[
            hidden_index
        ]
        coefficients = layer1.C[
            input_index,
            hidden_index
        ]
        strength = edge_strength(
            coefficients
        )
        width = (
            1.0
            + 2.0 * strength
        )
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linewidth=width,
            alpha=0.55,
            zorder=1
        )
        # Midpoint
        mid_x = (
            start[0] + end[0]
        ) / 2

        mid_y = (
            start[1] + end[1]
        ) / 2
        equation = format_polynomial(
            coefficients
        )
        ax.text(
            mid_x,
            mid_y,
            equation,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.75
            ),
            zorder=3
        )

# drawing layer 2 edge

layer2 = network.layers[1]
for hidden_index in range(
    layer2.n_inputs
):

    start = hidden_positions[
        hidden_index
    ]

    end = output_positions[0]

    coefficients = layer2.C[
        hidden_index,
        0
    ]

    strength = edge_strength(
        coefficients
    )

    width = (
        1.0
        + 2.0 * strength
    )

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linewidth=width,
        alpha=0.55,
        zorder=1
    )

    mid_x = (
        start[0] + end[0]
    ) / 2

    mid_y = (
        start[1] + end[1]
    ) / 2

    equation = format_polynomial(
        coefficients
    )

    ax.text(
        mid_x,
        mid_y,
        equation,
        fontsize=7,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.75
        ),
        zorder=3
    )

# drawing input nodes
for index, position in enumerate(
    input_positions
):

    ax.scatter(
        position[0],
        position[1],
        s=1800,
        zorder=5
    )

    ax.text(
        position[0],
        position[1],
        f"x{index + 1}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        zorder=6
    )

# hidden nodes
for index, position in enumerate(
    hidden_positions
):

    ax.scatter(
        position[0],
        position[1],
        s=1800,
        zorder=5
    )

    ax.text(
        position[0],
        position[1],
        f"h{index + 1}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        zorder=6
    )

# output nodes
output_position = output_positions[0]

ax.scatter(
    output_position[0],
    output_position[1],
    s=2000,
    zorder=5
)

ax.text(
    output_position[0],
    output_position[1],
    "ŷ",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    zorder=6
)

ax.text(
    0.0,
    0.98,
    "INPUT",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    1.0,
    0.98,
    "HIDDEN WEB",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    2.0,
    0.98,
    "OUTPUT",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

ax.set_title(
    "Kumo No Su\n"
    "Learned Polynomial",
    fontsize=20,
    pad=20
)

ax.set_xlim(
    -0.35,
    2.35
)

ax.set_ylim(
    0.0,
    1.05
)

ax.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print(
    f"Saved Kumo web to: "
    f"{OUTPUT_PATH}"
)

plt.show()