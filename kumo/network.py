import numpy as np

from kumo.layer import KumoLayer


class KumoNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(
        self,
        n_inputs,
        n_outputs,
        degree
    ):
        layer = KumoLayer(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            degree=degree
        )

        self.layers.append(
            layer
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def inspect(self, x):
        activations = [x]

        for layer in self.layers:
            x = layer.forward(x)

            activations.append(
                x.copy()
            )

        return activations

    def backward(self, error):
        gradients = []

        current_gradient = error

        for layer in reversed(
            self.layers
        ):

            (
                coefficient_gradients,
                bias_gradients,
                input_gradients
            ) = layer.backward(
                current_gradient
            )

            gradients.append(
                (
                    coefficient_gradients,
                    bias_gradients
                )
            )

            current_gradient = (
                input_gradients
            )

        gradients.reverse()

        return gradients

    def update(
        self,
        gradients,
        learning_rate
    ):
        for layer, gradients_for_layer in zip(
            self.layers,
            gradients
        ):

            (
                coefficient_gradients,
                bias_gradients
            ) = gradients_for_layer

            layer.update(
                coefficient_gradients,
                bias_gradients,
                learning_rate
            )

    def save(self, filename):
        data = {}

        data["format_version"] = np.array([
            2
        ])

        data["n_layers"] = np.array([
            len(self.layers)
        ])

        for i, layer in enumerate(
            self.layers
        ):

            data[f"layer_{i}_config"] = np.array([
                layer.n_inputs,
                layer.n_outputs,
                layer.degree
            ])

            data[f"layer_{i}_C"] = (
                layer.C
            )

            data[f"layer_{i}_b"] = (
                layer.b
            )

        np.savez(
            filename,
            **data
        )

    @classmethod
    def load(cls, filename):
        data = np.load(
            filename
        )

        network = cls()

        if "format_version" in data.files:
            format_version = int(
                data["format_version"][0]
            )
        else:
            format_version = 1

        n_layers = int(
            data["n_layers"][0]
        )

        for i in range(
            n_layers
        ):

            config = data[
                f"layer_{i}_config"
            ]

            n_inputs = int(
                config[0]
            )

            n_outputs = int(
                config[1]
            )

            degree = int(
                config[2]
            )

            network.add_layer(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                degree=degree
            )

            if format_version == 1:

                old_C = data[
                    f"layer_{i}_C"
                ]

                network.layers[i].b = np.sum(
                    old_C[:, :, 0],
                    axis=0
                )

                network.layers[i].C = (
                    old_C[:, :, 1:].copy()
                )

            elif format_version == 2:

                network.layers[i].C = data[
                    f"layer_{i}_C"
                ].copy()

                network.layers[i].b = data[
                    f"layer_{i}_b"
                ].copy()

            else:
                raise ValueError(
                    "Unsupported Kumo model "
                    f"format version: {format_version}"
                )

        return network