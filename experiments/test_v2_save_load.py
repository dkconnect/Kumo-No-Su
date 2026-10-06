import os
import numpy as np

from kumo.network import KumoNetwork


np.random.seed(42)

MODEL_PATH = "models/test_v2_kumo.npz"

network = KumoNetwork()

network.add_layer(
    n_inputs=2,
    n_outputs=3,
    degree=2
)

network.add_layer(
    n_inputs=3,
    n_outputs=1,
    degree=2
)

x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


for epoch in range(5000):

    prediction = network.forward(
        x
    )

    error = (
        prediction - y
    )

    gradients = network.backward(
        error
    )

    network.update(
        gradients,
        learning_rate=0.01
    )


before_save = network.forward(
    x
)

network.save(
    MODEL_PATH
)

loaded = KumoNetwork.load(
    MODEL_PATH
)

after_load = loaded.forward(
    x
)


print(
    "\nPredictions before save:"
)

print(
    before_save
)

print(
    "\nPredictions after load:"
)

print(
    after_load
)

print(
    "\nMaximum difference:",
    np.max(
        np.abs(
            before_save
            - after_load
        )
    )
)

print(
    "\nLoaded architecture:"
)

for i, layer in enumerate(
    loaded.layers
):

    print(
        f"Layer {i + 1}"
    )

    print(
        "C:",
        layer.C.shape
    )

    print(
        "b:",
        layer.b.shape
    )


data = np.load(
    MODEL_PATH
)

print(
    "\nModel format version:",
    int(
        data["format_version"][0]
    )
)

data.close()

os.remove(
    MODEL_PATH
)