import numpy as np
from kumo.network import KumoNetwork

network = KumoNetwork()

network.add_layer(
    n_inputs=2,
    n_outputs=3,
    degree=2
)

network.add_layer(
    n_inputs=3,
    n_outputs=1,
    degree=2
)

x = np.array([
    2.0,
    3.0
])

prediction = network.forward(x)

target = np.array([5.0])

error = prediction - target

gradients = network.backward(error)

print("Input:")
print(x)

print("\nPrediction:")
print(prediction)

print("\nNetwork layers:")
print(len(network.layers))

print("\nLayer 1 coefficients:")
print(network.layers[0].C.shape)

print("\nLayer 2 coefficients:")
print(network.layers[1].C.shape)

print("\nError:")
print(error)

print("\nNumber of gradient arrays:")
print(len(gradients))

print("\nLayer 1 gradient shape:")
print(gradients[0].shape)

print("\nLayer 2 gradient shape:")
print(gradients[1].shape)