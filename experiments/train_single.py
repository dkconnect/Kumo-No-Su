import numpy as np
from kumo.layer import KumoLayer

layer = KumoLayer(
    n_inputs=2,
    n_outputs=1,
    degree=2
)

x = np.array([2.0, 3.0])
target = np.array([5.0])

prediction = layer.forward(x)
error = prediction - target
loss = 0.5 * np.sum(error ** 2)

print("Input:", x)
print("Prediction:", prediction)
print("Target:", target)
print("Error:", error)
print("Loss:", loss)