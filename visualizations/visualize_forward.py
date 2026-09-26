import os
import numpy as np
import matplotlib.pyplot as plt
from kumo.network import KumoNetwork

MODEL_PATH = "models/xor_kumo.npz"
OUTPUT_PATH = "outputs/kumo_forward.png"
os.makedirs("outputs", exist_ok=True)

x = np.array([
    0.0,
    1.0
])

# trained kumo loading
network = KumoNetwork.load(
    MODEL_PATH
)

# polynomial fn.
def evaluate_edge(coefficients, value):
    """
    Calculating for one edge.
    """
    result = 0.0

    for degree, coefficient in enumerate(
        coefficients
    ):
        result += (
            coefficient * value ** degree
        )

    return result

# Manual forward pass
# Layer 1

layer1 = network.layers[0]
layer1_contributions = np.zeros(
    (
        layer1.n_inputs,
        layer1.n_outputs
    )
)

for input_index in range(
    layer1.n_inputs
):
    for hidden_index in range(
        layer1.n_outputs
    ):
        coefficients = layer1.C[
            input_index,
            hidden_index
        ]

        layer1_contributions[
            input_index,
            hidden_index
        ] = evaluate_edge(
            coefficients,
            x[input_index]
        )

hidden_values = np.sum(
    layer1_contributions,
    axis=0
)

# Layer 2
layer2 = network.layers[1]
layer2_contributions = np.zeros(
    (
        layer2.n_inputs,
        layer2.n_outputs
    )
)

for hidden_index in range(
    layer2.n_inputs
):

    coefficients = layer2.C[
        hidden_index,
        0
    ]

    layer2_contributions[
        hidden_index,
        0
    ] = evaluate_edge(
        coefficients,
        hidden_values[hidden_index]
    )

output_values = np.sum(
    layer2_contributions,
    axis=0
)

# verification
real_prediction = network.forward(x)

print("Input:")
print(x)

print("\nLayer 1 edge contributions:")
print(layer1_contributions)

print("\nHidden node values:")
print(hidden_values)

print("\nLayer 2 edge contributions:")
print(layer2_contributions)

print("\nVisualizer output:")
print(output_values)

print("\nKumo forward() output:")
print(real_prediction)

print(
    "\nDifference:",
    np.abs(
        output_values - real_prediction
    )
)

input_positions = [
    (0.0, 0.7),
    (0.0, 0.3)
]

hidden_positions = [
    (1.0, 0.8),
    (1.0, 0.5),
    (1.0, 0.2)
]

output_position = (
    2.0,
    0.5
)

# edge width 

def contribution_width(value):
    return (
        1.0
        + 4.0 * abs(value)
    )

fig, ax = plt.subplots(
    figsize=(14, 8)
)

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

        contribution = (
            layer1_contributions[
                input_index,
                hidden_index
            ]
        )

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linewidth=contribution_width(
                contribution
            ),
            alpha=0.55,
            zorder=1
        )

        mid_x = (
            start[0] + end[0]
        ) / 2

        mid_y = (
            start[1] + end[1]
        ) / 2

        ax.text(
            mid_x,
            mid_y,
            f"φ({x[input_index]:.2f})"
            f" = {contribution:+.3f}",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.85
            ),
            zorder=3
        )

for hidden_index in range(
    layer2.n_inputs
):

    start = hidden_positions[
        hidden_index
    ]

    end = output_position

    contribution = (
        layer2_contributions[
            hidden_index,
            0
        ]
    )

    hidden_value = hidden_values[
        hidden_index
    ]

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linewidth=contribution_width(
            contribution
        ),
        alpha=0.55,
        zorder=1
    )

    mid_x = (
        start[0] + end[0]
    ) / 2

    mid_y = (
        start[1] + end[1]
    ) / 2

    ax.text(
        mid_x,
        mid_y,
        f"φ({hidden_value:.2f})"
        f" = {contribution:+.3f}",
        fontsize=8,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.85
        ),
        zorder=3
    )

for index, position in enumerate(
    input_positions
):

    ax.scatter(
        position[0],
        position[1],
        s=2000,
        zorder=5
    )

    ax.text(
        position[0],
        position[1],
        f"x{index + 1}\n"
        f"{x[index]:.2f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=6
    )

for index, position in enumerate(
    hidden_positions
):

    ax.scatter(
        position[0],
        position[1],
        s=2200,
        zorder=5
    )

    ax.text(
        position[0],
        position[1],
        f"h{index + 1}\n"
        f"{hidden_values[index]:.3f}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        zorder=6
    )

ax.scatter(
    output_position[0],
    output_position[1],
    s=2400,
    zorder=5
)

ax.text(
    output_position[0],
    output_position[1],
    "ŷ\n"
    f"{output_values[0]:.4f}",
    ha="center",
    va="center",
    fontsize=12,
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
    "Kumo No Su — Live Forward Pass\n"
    f"Input = {x.tolist()}",
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
ax.axis("off")

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print(
    f"\nSaved visualization to: "
    f"{OUTPUT_PATH}"
)
plt.show()