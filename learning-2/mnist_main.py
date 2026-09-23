import numpy as np
from dataset_loader import load_mnist_pure_numpy
from kumo_network import KumoNoSu

print("Fetching Raw MNIST")
X_raw, y_raw = load_mnist_pure_numpy(num_samples=5000)

X_norm = (X_raw / 127.5) - 1.0

y_onehot = np.zeros((len(y_raw), 10))
y_onehot[np.arange(len(y_raw)), y_raw] = 1.0

split_idx = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]

print(f"Training set: {X_train.shape[0]} images")
print(f"Testing set:  {X_test.shape[0]} images")

print("\nInitializing KumoNoSu (Degree 2 Polynomial Edges)")
kumo_digit_clf = KumoNoSu(layer_sizes=[784, 64, 10], degree=2)

print("\nTraining KumoNoSu on Handwritten Digits")
kumo_digit_clf.fit_mnist(X_train, y_train, epochs=30, batch_size=128, lr=0.2)

# Test Evaluation
test_logits = kumo_digit_clf.forward(X_test)
test_probs = kumo_digit_clf.softmax(test_logits)
test_preds = np.argmax(test_probs, axis=1)
test_targets = np.argmax(y_test, axis=1)

final_acc = np.mean(test_preds == test_targets) * 100
print("\n" + "=" * 45)
print(f" FINAL TEST SET ACCURACY: {final_acc:.2f}%")
print("=" * 45)