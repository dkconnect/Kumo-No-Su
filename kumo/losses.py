import numpy as np

def softmax(logits):
    """ Converts raw network outputs into probabilities """

    logits = np.asarray(
        logits,
        dtype=float
    )

    single_input = (
        logits.ndim == 1
    )

    if single_input:
        logits = logits[None, :]

    if logits.ndim != 2:
        raise ValueError(
            "Softmax input must have shape (classes,) or (batch, classes)"
        )

    # NUMERICAL STABILITY
    shifted_logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exponentials = np.exp(
        shifted_logits
    )

    probabilities = (
        exponentials
        / np.sum(
            exponentials,
            axis=1,
            keepdims=True
        )
    )

    if single_input:
        return probabilities[0]
    return probabilities


def cross_entropy(
    probabilities,
    targets
):
    """ Mean cross-entropy loss """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    targets = np.asarray(
        targets,
        dtype=float
    )

    if probabilities.ndim == 1:
        probabilities = (
            probabilities[None, :]
        )

    if targets.ndim == 1:
        targets = targets[None, :]

    if probabilities.shape != targets.shape:
        raise ValueError(
            "Probabilities and targets must have the same shape"
        )

    # Prevent log(0)
    epsilon = 1e-12

    safe_probabilities = np.clip(
        probabilities,
        epsilon,
        1.0
    )

    sample_losses = -np.sum(
        targets
        * np.log(safe_probabilities),
        axis=1
    )

    return np.mean(
        sample_losses
    )

def softmax_cross_entropy_gradient(
    probabilities,
    targets
):
    """ Gradient of softmax + cross-entropy wrt the logits """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    targets = np.asarray(
        targets,
        dtype=float
    )

    if probabilities.shape != targets.shape:
        raise ValueError(
            "Probabilities and targets must have the same shape"
        )

    return (
        probabilities
        - targets
    )