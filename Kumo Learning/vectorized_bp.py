import numpy as np

X = np.array([[2.0, 3.0]]) 
y_true = np.array([[1.0]])  

np.random.seed(42)
W = np.array([[0.5], [-0.2]])     
b = np.array([[0.1]])       
alpha = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.dot(X, W) + b  
y_pred = sigmoid(z)
loss = np.mean((y_pred - y_true) ** 2)

dz = 2 * (y_pred - y_true) * (y_pred * (1 - y_pred)) 

dW = np.dot(X.T, dz)
db = dz 

W_new = W - (alpha * dW)
b_new = b - (alpha * db)

print(f"Prediction: {y_pred[0,0]:.4f}")
print(f"Weight Gradients (dW):\n{dW}")
print(f"Updated Weights (W_new):\n{W_new}")