import numpy as np
from kumo_network import KumoNoSu
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

print("Loading MNIST Digits Dataset")
X_raw, y_raw = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False
)

X_subset = X_raw[:5000]
y_subset = y_raw[:5000].astype(int)

X_norm = (X_subset / 127.5) - 1.0

y_onehot = np.zeros((len(y_subset), 10))
y_onehot[np.arange(len(y_subset)), y_subset] = 1.0

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_onehot, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Testing set:  {X_test.shape[0]} images")

print("\n--- Initializing KumoNoSu (Degree 2 Polynomial Edges) ---")
kumo_digit_clf = KumoNoSu(layer_sizes=[784, 32, 10], degree=2)

print("\n--- Training KumoNoSu on Handwritten Digits ---")
kumo_digit_clf.fit_mnist(X_train, y_train, epochs=15, batch_size=128, lr=0.05)

test_logits = kumo_digit_clf.forward(X_test)
test_probs = kumo_digit_clf.softmax(test_logits)
test_preds = np.argmax(test_probs, axis=1)
test_targets = np.argmax(y_test, axis=1)

final_acc = np.mean(test_preds == test_targets) * 100
print("\n" + "=" * 45)
print(f" FINAL TEST SET ACCURACY: {final_acc:.2f}%")
print("=" * 45)