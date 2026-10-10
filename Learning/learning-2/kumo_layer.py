import numpy as np

class PolynomialBasis:
    """Expands inputs into polynomial power tensors [x^0, x^1, x^2, ... x^d]."""

    def __init__(self, degree=3):
        self.degree = degree
        self.D = degree + 1

    def forward(self, X):
        batch_size, in_features = X.shape
        B = np.ones((batch_size, in_features, self.D))
        for d in range(1, self.D):
            B[:, :, d] = X**d
        return B

    def derivative(self, X):
        batch_size, in_features = X.shape
        dB_dX = np.zeros((batch_size, in_features, self.D))
        for d in range(1, self.D):
            dB_dX[:, :, d] = d * (X ** (d - 1))
        return dB_dX


class KumoLayer:
    """Edge-based Learnable Polynomial Curve Layer with L1 Regularization."""

    def __init__(self, in_features, out_features, degree=3):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.D = degree + 1
        self.basis = PolynomialBasis(degree)

        # Tensor
        self.C = (
            np.random.randn(in_features, out_features, self.D)
            * np.sqrt(2.0 / in_features)
            * 0.1
        )
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        self.X_in = X
        self.B = self.basis.forward(X)
        Y = np.einsum("nid,iod->no", self.B, self.C) + self.b
        return Y

    def backward(self, dL_dY, lr, l1_lambda=1e-4):
        batch_size = dL_dY.shape[0]

        # Base Coefficient
        dL_dC = np.einsum("nid,no->iod", self.B, dL_dY) / batch_size
        dL_db = np.sum(dL_dY, axis=0, keepdims=True) / batch_size

        # Regularization Gradient: l1_lambda * sign(C)
        dL_dC += l1_lambda * np.sign(self.C)

        # Backpropagate error 
        dL_dB = np.einsum("no,iod->nid", dL_dY, self.C)

        # Basis Derivative wrt X
        dB_dX = self.basis.derivative(self.X_in)

        # Chain Rule
        dL_dX = np.sum(dL_dB * dB_dX, axis=2)

        self.C -= lr * dL_dC
        self.b -= lr * dL_db

        return dL_dX