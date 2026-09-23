import gzip
import os
import urllib.request
import numpy as np

def load_mnist_pure_numpy(num_samples=5000):
    # Downloads and parses raw MNIST
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "X": "train-images-idx3-ubyte.gz",
        "y": "train-labels-idx1-ubyte.gz",
    }

    data = {}
    for key, filename in files.items():
        if not os.path.exists(filename):
            urllib.request.urlretrieve(base_url + filename, filename)

        with gzip.open(filename, "rb") as f:
            if key == "X":
                f.read(16)
                buf = f.read(num_samples * 28 * 28)
                images = (
                    np.frombuffer(buf, dtype=np.uint8)
                    .reshape(num_samples, 784)
                    .astype(np.float32)
                )
                data["X"] = images
            else:
                f.read(8)
                buf = f.read(num_samples)
                labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
                data["y"] = labels

    return data["X"], data["y"]