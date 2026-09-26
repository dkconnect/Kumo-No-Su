import numpy as np

class KumoLayer:
    def __init__(self, n_inputs, n_outputs, degree):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.degree = degree
        self.n_coeffs = degree + 1

        # I,O,C
        self.C = np.random.randn(
            n_inputs,
            n_outputs,
            self.n_coeffs
        ) * 0.1

        self.x = None
        self.powers = None

    def forward(self, x):
        self.x = x

        # I,1,C
        self.powers = (
            x[:, None, None]
            ** np.arange(self.n_coeffs)
        )

        # I,O,C
        terms = self.C * self.powers

        # polynomial coeff.
        edge_outputs = np.sum(
            terms,
            axis=2
        )

        output = np.sum(
            edge_outputs,
            axis=0
        )

        return output

    def backward(self, error):
        # COEFFICIENT GRADIENTS

        # so it broadcasts across inputs nd polynomial coefficients.

        error_reshaped = error[None, :, None]

        # (I,1,C) x (1,O,1) = (I,O,C)

        coefficient_gradients = (
            self.powers * error_reshaped
        )

        # INPUT GRADIENTS

        degrees = np.arange(
            self.n_coeffs
        )

        power_derivatives = np.zeros_like(
            self.powers
        )

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

        # error shape
        # sum across outputs

        input_gradients = np.sum(
            edge_derivatives * error,
            axis=1
        )

        return (
            coefficient_gradients,
            input_gradients
        )

    def update(
        self,
        coefficient_gradients,
        learning_rate
    ):
        self.C -= (
            learning_rate
            * coefficient_gradients
        )