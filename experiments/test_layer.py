import numpy as np
from kumo.layer import KumoLayer

layer = KumoLayer(
    n_inputs=2,
    n_outputs=3,
    degree=2
)

x = np.array([2.0, 3.0])
y = layer.forward(x)

print("Input:", x)
print("Output:", y)