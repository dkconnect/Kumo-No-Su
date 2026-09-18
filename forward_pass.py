import numpy as np

N = 4  
I = 2  
O = 1  
D = 4 

np.random.seed(42)

B = np.random.randn(N, I, D)
Y_true = np.array([[1.0], [0.0], [1.0], [0.0]])

C = np.random.randn(I, O, D) * 0.1

Y_pred = np.einsum('nid,iod->no', B, C)

loss = np.mean((Y_pred - Y_true) ** 2)

# BACKWARD PASS
dL_dY = 2 * (Y_pred - Y_true)         

dL_dC = np.einsum('nid,no->iod', B, dL_dY) / N

print(f"Loss: {loss:.4f}")
print("Coefficient Gradient Shape (dC):", dL_dC.shape)
print("Computed Gradient Tensor dC:\n", dL_dC)