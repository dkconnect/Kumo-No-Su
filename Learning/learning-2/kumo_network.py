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

    def softmax(self, logits):
        exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self, dL_dPred, lr, l1_lambda=1e-5):
        grad = dL_dPred
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr, l1_lambda=l1_lambda)

    def fit_mnist(self, X, y_onehot, epochs=20, batch_size=128, lr=0.01):
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            total_loss = 0.0
            correct = 0

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                logits = self.forward(X_batch)
                probs = self.softmax(logits)

                loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))
                total_loss += loss * len(X_batch)

                preds = np.argmax(probs, axis=1)
                targets = np.argmax(y_batch, axis=1)
                correct += np.sum(preds == targets)
                dL_dLogits = (probs - y_batch) / len(X_batch)

                # Backward pass
                self.backward(dL_dLogits, lr=lr)

            acc = (correct / num_samples) * 100
            avg_loss = total_loss / num_samples
            print(
                f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%"
            )