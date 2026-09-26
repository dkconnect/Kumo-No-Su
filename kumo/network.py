import numpy as np
from kumo.layer import KumoLayer

class KumoNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, n_inputs, n_outputs, degree):
        layer = KumoLayer(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            degree=degree
        )

        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, error):
        gradients = []

        current_gradient = error
        for layer in reversed(self.layers):

            coefficient_gradients, input_gradients = (
                layer.backward(current_gradient)
            )

            gradients.append(
                coefficient_gradients
            )

            current_gradient = input_gradients

        gradients.reverse()
        return gradients

    def update(self, gradients, learning_rate):
        for layer, coefficient_gradients in zip(
            self.layers,
            gradients
        ):
            layer.update(
                coefficient_gradients,
                learning_rate
            )

    def save(self, filename):
        data = {}

        data["n_layers"] = np.array(
            [len(self.layers)]
        )

        # arch. + coefficients
        # for every layer
        for i, layer in enumerate(self.layers):

            data[f"layer_{i}_config"] = np.array([
                layer.n_inputs,
                layer.n_outputs,
                layer.degree
            ])

            data[f"layer_{i}_C"] = layer.C

        np.savez(
            filename,
            **data
        )

    @classmethod
    def load(cls, filename):
        data = np.load(filename)

        network = cls()
        n_layers = int(
            data["n_layers"][0]
        )

        for i in range(n_layers):
            config = data[
                f"layer_{i}_config"
            ]

            n_inputs = int(config[0])
            n_outputs = int(config[1])
            degree = int(config[2])

            network.add_layer(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                degree=degree
            )

            network.layers[i].C = (
                data[f"layer_{i}_C"].copy()
            )

        return network