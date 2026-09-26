import numpy as np
from kumo.network import KumoNetwork

network = KumoNetwork.load(
    "models/xor_kumo.npz"
)

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

print("Loaded Kumo predictions:")

for x in X:
    prediction = network.forward(x)
    print(
        f"{x} -> "
        f"{prediction[0]:.6f}"
    )

print("\nLoaded architecture:")
for i, layer in enumerate(network.layers):

    print(
        f"Layer {i + 1}: "
        f"{layer.n_inputs} -> "
        f"{layer.n_outputs}, "
        f"degree {layer.degree}"
    )