import numpy as np
from kumo_network import KumoNoSu

np.random.seed(42)
X_train = np.random.uniform(-1.5, 1.5, (500, 2))

# non-linear combination
y_train = np.sin(X_train[:, 0:1]) + (X_train[:, 1:2] ** 2)

# Network
# 2 Inputs -> 4 Hidden Nodes -> 1 Output
model = KumoNoSu(layer_sizes=[2, 4, 1], degree=3)

print("--- Training Kumo No Su (蜘蛛の巣) ---")
model.fit(X_train, y_train, epochs=3000, lr=0.01)

X_test = np.array([[0.5, 1.0], [-1.0, 0.5]])

y_true = np.sin(X_test[:, 0:1]) + (X_test[:, 1:2] ** 2)
y_pred = model.forward(X_test)

print("\n--- Model Verification ---")
for i in range(len(X_test)):
    print(f"Input: {X_test[i]}")
    print(f"  Target Output:    {y_true[i][0]:.4f}")
    print(f"  Kumo Prediction:  {y_pred[i][0]:.4f}\n")