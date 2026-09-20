import numpy as np
from kumo_network import KumoNoSu
from symbolic_exporter import extract_formula

np.random.seed(42)
X_train = np.random.uniform(-1.5, 1.5, (1000, 2))
y_train = np.sin(X_train[:, 0:1]) + (X_train[:, 1:2] ** 2)

# Single KumoLayer
model = KumoNoSu(layer_sizes=[2, 1], degree=3)

print("--- Training Kumo No Su for Formula Extraction ---")
model.fit(X_train, y_train, epochs=4000, lr=0.02, l1_lambda=1e-4)

extract_formula(model.layers[0], threshold=0.02)