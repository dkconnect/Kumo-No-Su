import numpy as np


def print_ascii_digit(image_flat, true_label, pred_label):
    img = image_flat.reshape(28, 28)
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]

    print("\n" + "=" * 32)
    print(f"TRUE LABEL: {true_label}  |  KUMO PRED: {pred_label}")
    print("=" * 32)

    for row in img:
        line = ""
        for pixel in row:
            idx = int(((pixel + 1.0) / 2.0) * 9)
            idx = min(max(idx, 0), 9)
            line += chars[idx]
        print(line)
    print("=" * 32)


def inspect_predictions(model, X_test, y_test, num_samples=5):
    logits = model.forward(X_test[:num_samples])
    probs = model.softmax(logits)
    preds = np.argmax(probs, axis=1)
    targets = np.argmax(y_test[:num_samples], axis=1)

    for i in range(num_samples):
        print_ascii_digit(X_test[i], targets[i], preds[i])
        print("Class Probabilities:")
        for digit in range(10):
            bar = "█" * int(probs[i, digit] * 20)
            print(f"  Digit {digit}: {probs[i, digit]*100:5.1f}% | {bar}")