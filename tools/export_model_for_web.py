import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kumo.network import KumoNetwork

MODEL_PATH = "models/mnist_full_kumo.npz"
OUTPUT_PATH = "web/model/kumo_mnist.json"


network = KumoNetwork.load(
    MODEL_PATH
)

model_data = {
    "format": "kumo-no-su",
    "version": 2,
    "layers": []
}

for layer in network.layers:
    layer_data = {
        "n_inputs": layer.n_inputs,
        "n_outputs": layer.n_outputs,
        "degree": layer.degree,
        "coefficients": layer.C.tolist(),
        "biases": layer.b.tolist()
    }

    model_data["layers"].append(
        layer_data
    )

output_path = Path(
    OUTPUT_PATH
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        model_data,
        file
    )

print(
    "Exported Kumo model:"
)

print(
    MODEL_PATH,
    "->",
    OUTPUT_PATH
)

print(
    "\nLayers:"
)

for i, layer in enumerate(
    network.layers
):
    print(
        f"Layer {i + 1}:",
        f"{layer.n_inputs} -> {layer.n_outputs}",
        f"degree {layer.degree}"
    )