from data.load_mnist import load_mnist_split

X_train, y_train, X_test, y_test = (load_mnist_split())

print("\nOfficial training portion:")
print("Images:", X_train.shape)
print("Labels:", y_train.shape)

print("\nOfficial test portion:")
print("Images:", X_test.shape)
print("Labels:", y_test.shape)

print("\nPixel range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())