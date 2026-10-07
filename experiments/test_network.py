import numpy as np

from kumo.network import KumoNetwork


np.random.seed(42)

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

target = np.array([
    5.0
])

prediction = network.forward(x)

error = prediction - target

gradients = network.backward(error)

print("Input:")
print(x)

print("\nPrediction:")
print(prediction)

print("\nNetwork layers:")
print(len(network.layers))

for i, layer in enumerate(network.layers):
    print(f"\nLayer {i + 1} coefficients:")
    print(layer.C.shape)

    print(f"Layer {i + 1} bias:")
    print(layer.b.shape)

print("\nError:")
print(error)

print("\nNumber of gradient groups:")
print(len(gradients))

for i, (
    coefficient_gradients,
    bias_gradients
) in enumerate(gradients):

    print(
        f"\nLayer {i + 1} "
        "coefficient gradient shape:"
    )
    print(
        coefficient_gradients.shape
    )

    print(
        f"Layer {i + 1} "
        "bias gradient shape:"
    )
    print(
        bias_gradients.shape
    )