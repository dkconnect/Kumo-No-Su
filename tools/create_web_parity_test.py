import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kumo.network import KumoNetwork
from kumo.losses import softmax


MODEL_PATH = ROOT / "models" / "mnist_full_kumo.npz"
OUTPUT_PATH = ROOT / "web" / "model" / "parity_test.json"


def main():
    rng = np.random.default_rng(42)

    input_data = rng.random(
        (1, 784)
    )

    network = KumoNetwork.load(
        MODEL_PATH
    )

    activations = network.inspect(
        input_data
    )

    hidden = activations[1][0]
    logits = activations[2][0]
    probabilities = softmax(
        logits.reshape(1, -1)
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    fixture = {
        "input": input_data[0].tolist(),
        "hidden": hidden.tolist(),
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "prediction": prediction
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            fixture,
            file,
            indent=2
        )

    print(
        "Created web parity fixture:"
    )

    print(
        OUTPUT_PATH.relative_to(ROOT)
    )

    print()

    print(
        f"Prediction: {prediction}"
    )

    print(
        "Logits:"
    )

    print(
        logits
    )

    print(
        "Probabilities:"
    )

    print(
        probabilities
    )


if __name__ == "__main__":
    main()