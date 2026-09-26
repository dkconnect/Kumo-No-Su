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

        # starting gradient coming
        # from the loss.
        current_gradient = error

        for layer in reversed(self.layers):

            coefficient_gradients, input_gradients = (
                layer.backward(current_gradient)
            )

            gradients.append(coefficient_gradients)

            current_gradient = input_gradients

        # Layer 2 gradient
        # Layer 1 gradient
        gradients.reverse()
        return gradients