import numpy as np

# Sigmoid & Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

input_dim = 2
hidden_dim = 3
output_dim = 1

np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim)  
b1 = np.zeros((1, hidden_dim))               

W2 = np.random.randn(hidden_dim, output_dim) 
b2 = np.zeros((1, output_dim))               

lr = 1.0
epochs = 5000

for epoch in range(epochs):
    Z1 = np.dot(X, W1) + b1 
    A1 = sigmoid(Z1)   
    
    Z2 = np.dot(A1, W2) + b2  
    A2 = sigmoid(Z2)       
    
    loss = np.mean((A2 - Y) ** 2)
 
    dZ2 = 2 * (A2 - Y) * sigmoid_derivative(A2) 
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]       
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
    
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1) 
    dW1 = np.dot(X.T, dZ1) / X.shape[0]       
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
    
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print("--- Predictions after training ---")
print(np.round(A2, 3))