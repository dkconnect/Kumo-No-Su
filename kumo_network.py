import numpy as np
from kumo_layer import KumoLayer


class KumoNoSu:
    def __init__(self, layer_sizes, degree=3):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                KumoLayer(layer_sizes[i], layer_sizes[i + 1], degree=degree)
            )

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dL_dPred, lr, l1_lambda=1e-4):
        grad = dL_dPred
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr, l1_lambda=l1_lambda)

    def fit(self, X, y, epochs=3000, lr=0.01, l1_lambda=1e-4):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y) ** 2)
            dL_dPred = 2 * (y_pred - y)

            self.backward(dL_dPred, lr, l1_lambda=l1_lambda)

            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d} | MSE Loss: {loss:.6f}")