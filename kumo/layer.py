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

        self.single_input = False

    # FORWARD
    def forward(self, x):

        x = np.asarray(
            x,
            dtype=float
        )

        # single sample(I) batch to 1,I 
        self.single_input = (
            x.ndim == 1
        )

        if self.single_input:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError(
                "KumoLayer input must have shape "
                "(inputs,) or (batch, inputs)"
            )

        if x.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} inputs, "
                f"received {x.shape[1]}"
            )

        self.x = x

        # POLYNOMIAL POWERS
        degrees = np.arange(
            self.n_coeffs
        )

        self.powers = (
            x[:, :, None, None]
            ** degrees
        )

        # applying edge fn.
        terms = (
            self.powers
            * self.C[None, :, :, :]
        )

        # Sum polynomial terms.
        edge_outputs = np.sum(
            terms,
            axis=3
        )

        # Sum all incoming edges.
        output = np.sum(
            edge_outputs,
            axis=1
        )

        if self.single_input:
            return output[0]

        return output

    # backward
    def backward(self, error):
        error = np.asarray(
            error,
            dtype=float
        )

        if error.ndim == 1:
            error = error[None, :]

        if error.ndim != 2:
            raise ValueError(
                "KumoLayer error must have shape "
                "(outputs,) or (batch, outputs)"
            )

        if error.shape[0] != self.x.shape[0]:
            raise ValueError(
                "Error batch size does not match "
                "the previous forward pass"
            )

        if error.shape[1] != self.n_outputs:
            raise ValueError(
                f"Expected {self.n_outputs} output "
                f"gradients, received {error.shape[1]}"
            )

        # coeff. gradients 
        error_expanded = (
            error[:, None, :, None]
        )

        sample_coefficient_gradients = (
            self.powers
            * error_expanded
        )

        # Average across the complete batch.
        # I,O,C

        coefficient_gradients = np.mean(
            sample_coefficient_gradients,
            axis=0
        )

        # input gradients
        degrees = np.arange(
            self.n_coeffs
        )

        power_derivatives = np.zeros_like(
            self.powers
        )

        if self.n_coeffs > 1:
            power_derivatives[
                :, :, :, 1:
            ] = (
                degrees[1:]
                * self.x[:, :, None, None]
                ** (degrees[1:] - 1)
            )


        # Multiply polynomial coefficients by power derivatives.
        edge_derivative_terms = (
            power_derivatives
            * self.C[None, :, :, :]
        )


        # Sum coefficient dimn.
        edge_derivatives = np.sum(
            edge_derivative_terms,
            axis=3
        )
        input_gradient_contributions = (
            edge_derivatives
            * error[:, None, :]
        )

        # Each input connects to every output,
        input_gradients = np.sum(
            input_gradient_contributions,
            axis=2
        )

        if self.single_input:
            input_gradients = (
                input_gradients[0]
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