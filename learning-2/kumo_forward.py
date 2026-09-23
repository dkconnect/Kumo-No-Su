import numpy as np

x = np.array([2, 3])

C = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

powers = x[:, None] ** np.arange(3)

terms = C * powers

edge_outputs = np.sum(terms, axis=1)

y1 = np.sum(edge_outputs)

print(y1)