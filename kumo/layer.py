import numpy as np

class KumoLayer:
    def __init__(self, n_inputs, n_outputs, degree):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.degree = degree

        # Degree 2: c0 + c1*x + c2*x^2
        # so we need 3 coefficients

        self.n_coeffs = degree + 1

        self.C = np.random.randn(
            n_inputs,
            n_outputs,
            self.n_coeffs
        ) * 0.1

        self.x = None
        self.powers = None

    def forward(self, x):
        self.x = x
        self.powers = (
            x[:, None, None]
            ** np.arange(self.n_coeffs)
        )

        terms = self.C * self.powers

        # Sum polynomial terms.
        edge_outputs = np.sum(
            terms,
            axis=2
        )

        # Sum all incoming edges for each output.
        output = np.sum(
            edge_outputs,
            axis=0
        )

        return output

    def backward(self, error):
        # COEFFICIENT GRADIENTS
        coefficient_gradients = (
            self.powers * error
        )

        # INPUT GRADIENTS
        # We also need to know how the loss changes wrt each input.

        degrees = np.arange(self.n_coeffs)
        power_derivatives = np.zeros_like(
            self.powers
        )

        # Derivatives:

        if self.n_coeffs > 1:
            power_derivatives[:, :, 1:] = (
                degrees[1:]
                * self.x[:, None, None]
                ** (degrees[1:] - 1)
            )

        edge_derivatives = np.sum(
            self.C * power_derivatives,
            axis=2
        )

        # Each input may connect to multiple  outputs, so we sum the gradient coming backward through all of them.

        input_gradients = np.sum(
            edge_derivatives * error,
            axis=1
        )

        return coefficient_gradients, input_gradients

    def update(self, coefficient_gradients, learning_rate):

        self.C -= (
            learning_rate
            * coefficient_gradients
        )