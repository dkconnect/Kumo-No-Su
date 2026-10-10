import numpy as np

class DenseLayer:
    """A fully-connected neural network layer with Sigmoid activation."""
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        self.X_in = X  
        self.Z = np.dot(X, self.W) + self.b
        self.A = self.sigmoid(self.Z)
        return self.A

    def backward(self, dL_dA, lr):
        batch_size = dL_dA.shape[0]
        dZ = dL_dA * self.sigmoid_derivative(self.A)
        
        dW = np.dot(self.X_in.T, dZ) / batch_size
        db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        
        dL_dX = np.dot(dZ, self.W.T)
        
        self.W -= lr * dW
        self.b -= lr * db
        return dL_dX

class StandardMLP:
    """Sequential Multi-Layer Perceptron container."""
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dL_dPred, lr):
        grad = dL_dPred
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def fit(self, X, y, epochs=5000, lr=1.0):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            
            loss = np.mean((y_pred - y) ** 2)
            dL_dPred = 2 * (y_pred - y)
            
            self.backward(dL_dPred, lr)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")


# Testing Modular Network on XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = StandardMLP(layer_sizes=[2, 4, 4, 1])

print("Training Modular MLP...")
model.fit(X, y, epochs=5000, lr=1.0)

print("\nFinal Predictions:")
print(np.round(model.forward(X), 3))