import numpy as np

basis_vector = np.array([1.0, 2.0, 4.0, 8.0])

coefficients = np.array([0.5, 1.0, -0.5, 0.1])

edge_output = np.dot(basis_vector, coefficients)

print("Edge Output:", edge_output)