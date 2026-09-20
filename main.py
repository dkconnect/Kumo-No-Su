import numpy as np
from kumo_network import KumoNoSu
from visualize_edges import plot_edge_curves

np.random.seed(42)
X_train = np.random.uniform(-1.5, 1.5, (500, 2))
y_train = np.sin(X_train[:, 0:1]) + (X_train[:, 1:2] ** 2)

model = KumoNoSu(layer_sizes=[2, 4, 1], degree=3)

print("--- Training Kumo No Su (蜘蛛の巣) ---")
model.fit(X_train, y_train, epochs=3000, lr=0.01)

print("\n--- Plotting Learned Edge Curves ---")
plot_edge_curves(model.layers[0], layer_idx=0)
plot_edge_curves(model.layers[1], layer_idx=1)