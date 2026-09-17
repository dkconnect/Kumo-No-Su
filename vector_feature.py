import numpy as np

def polynomial_expansion(x, degree=3):
    batch_size = x.shape[0]
    in_features = x.shape[1]
    
    basis = np.ones((batch_size, in_features, degree + 1))

    for d in range(1, degree + 1):
        basis[:, :, d] = x ** d
    return basis

X = np.array([[3.0]]) 
X_expanded = polynomial_expansion(X, degree=3)

print("Original Input X:", X)
print("Expanded Basis Vector:", X_expanded)