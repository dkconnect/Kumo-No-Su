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