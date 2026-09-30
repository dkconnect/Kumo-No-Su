import numpy as np

from kumo.losses import (
    softmax,
    cross_entropy,
    softmax_cross_entropy_gradient
)

# fake KUMO output
logits = np.array([
    1.2,
    0.3,
    2.1
])

# Correct class is class 2.
target = np.array([
    0.0,
    0.0,
    1.0
])

# softmax
probabilities = softmax(
    logits
)

print()
print("Softmax Test")
print()

print("Logits:")

print(logits)

print()

print("Probabilities:")

print(probabilities)

print()

print("Probability sum:", np.sum(probabilities))

# CROSS ENTROPY
loss = cross_entropy(
    probabilities,
    target
)

print()

print("Cross-entropy loss:", loss
)

gradient = (
    softmax_cross_entropy_gradient(
        probabilities,
        target
    )
)

print()

print("Gradient:")

print(gradient)

#basic checks
print()
print("Checks")

probability_sum_correct = np.isclose(
    np.sum(probabilities),
    1.0
)

probabilities_positive = np.all(
    probabilities >= 0.0
)

gradient_sum_zero = np.isclose(
    np.sum(gradient),
    0.0
)

print("Probabilities sum to 1:", probability_sum_correct)

print("Probabilities non-negative:", probabilities_positive)

print("Gradient sums to 0:", gradient_sum_zero)

print()

if (
    probability_sum_correct
    and probabilities_positive
    and gradient_sum_zero
):
    print("PASS: Softmax and cross-entropy behave correctly.")

else:
    print("FAIL: Classification math needs inspection.")