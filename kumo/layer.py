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

        self.bias = np.zeros(n_outputs)

    def forward(self, x):
        powers = x[:, None, None] ** np.arange(self.n_coeffs)
        terms = self.C * powers
        edge_outputs = np.sum(terms, axis=2)
        output = np.sum(edge_outputs, axis=0) + self.bias

        print("edge outputs:")
        print(edge_outputs)
        print("final output:")
        print(output)

        return output