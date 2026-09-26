import numpy as np
import matplotlib.pyplot as plt
from kumo.network import KumoNetwork

network = KumoNetwork.load(
    "models/xor_kumo.npz"
)

def evaluate_polynomial(coefficients, x):
    y = np.zeros_like(
        x,
        dtype=float
    )

    for degree, coefficient in enumerate(
        coefficients
    ):
        y += coefficient * (x ** degree)

    return y

print("Learned Kumo edge functions:\n")

for layer_index, layer in enumerate(
    network.layers
):
    print(
        f"Layer {layer_index + 1}"
    )

    for input_index in range(
        layer.n_inputs
    ):

        for output_index in range(
            layer.n_outputs
        ):

            coefficients = layer.C[
                input_index,
                output_index
            ]

            terms = []

            for degree, coefficient in enumerate(
                coefficients
            ):
                terms.append(
                    f"{coefficient:+.4f}x^{degree}"
                )
            equation = " ".join(terms)

            print(
                f"  Edge "
                f"{input_index + 1} -> "
                f"{output_index + 1}: "
                f"φ(x) = {equation}"
            )

    print()


# plotting every edge fn.

for layer_index, layer in enumerate(
    network.layers
):

    n_edges = (
        layer.n_inputs
        * layer.n_outputs
    )

    fig, axes = plt.subplots(
        n_edges,
        1,
        figsize=(7, 3 * n_edges)
    )

    if n_edges == 1:
        axes = [axes]

    edge_number = 0

    for input_index in range(
        layer.n_inputs
    ):

        for output_index in range(
            layer.n_outputs
        ):

            ax = axes[edge_number]

            coefficients = layer.C[
                input_index,
                output_index
            ]

            x = np.linspace(
                -0.5,
                1.5,
                300
            )

            y = evaluate_polynomial(
                coefficients,
                x
            )

            ax.plot(
                x,
                y,
                linewidth=2
            )

            ax.axhline(
                0,
                linewidth=0.8
            )

            ax.axvline(
                0,
                linewidth=0.8
            )

            ax.grid(
                alpha=0.25
            )

            ax.set_title(
                f"Layer {layer_index + 1} | "
                f"Input {input_index + 1} "
                f"→ Output {output_index + 1}"
            )

            ax.set_xlabel("x")
            ax.set_ylabel("φ(x)")

            edge_number += 1

    fig.suptitle(
        f"Kumo No Su — "
        f"Layer {layer_index + 1} "
        f"Learned Edge Functions",
        fontsize=16
    )

    fig.tight_layout()

    output_filename = (
        f"layer_{layer_index + 1}_edges.png"
    )

    fig.savefig(
        output_filename,
        dpi=150
    )

    print(
        f"Saved: {output_filename}"
    )
plt.show()