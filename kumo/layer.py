import numpy as np


class KumoLayer:
    def __init__(self, n_inputs, n_outputs, degree):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.degree = degree

        self.n_coeffs = degree + 1

        self.C = np.random.randn(
            n_inputs,
            n_outputs,
            self.n_coeffs
        ) * 0.1

    def forward(self, x):
        powers = x[:, None, None] ** np.arange(self.n_coeffs)

        print("x shape:", x.shape)
        print("powers shape:", powers.shape)