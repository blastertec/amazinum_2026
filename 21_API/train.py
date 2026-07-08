"""Download MNIST, train an MLP classifier and save it to model.joblib."""

import gzip
import struct
import urllib.request
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

# Original MNIST hosting is unreliable, this GitHub mirror keeps the same files.
MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def download():
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        path = DATA_DIR / name
        if not path.exists():
            print("downloading", name)
            urllib.request.urlretrieve(MIRROR + name, path)


def read_images(path):
    # idx format: 16-byte header (magic, count, rows, cols), then raw pixels
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return pixels.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_labels(path):
    with gzip.open(path, "rb") as f:
        f.read(8)  # skip header
        return np.frombuffer(f.read(), dtype=np.uint8)


def main():
    download()
    x_train = read_images(DATA_DIR / FILES[0])
    y_train = read_labels(DATA_DIR / FILES[1])
    x_test = read_images(DATA_DIR / FILES[2])
    y_test = read_labels(DATA_DIR / FILES[3])
    print(f"train: {x_train.shape}, test: {x_test.shape}")

    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        batch_size=256,
        max_iter=15,
        random_state=0,
        verbose=True,
    )
    clf.fit(x_train, y_train)

    acc = clf.score(x_test, y_test)
    print(f"test accuracy: {acc:.4f}")

    joblib.dump(clf, MODEL_PATH)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
