import matplotlib.pyplot as plt
from data.load_mnist import load_mnist

X, y = load_mnist()
index = 2

flat_image = X[index]
label = y[index]

print("\nFlat image shape:")
print(flat_image.shape)

print("\nLabel:")
print(label)

image = flat_image.reshape(
    28,
    28
)

print("\nReshaped image:")
print(image.shape)

plt.figure(
    figsize=(5, 5)
)

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    f"MNIST digit: {label}"
)

plt.axis("off")

plt.tight_layout()
plt.show()