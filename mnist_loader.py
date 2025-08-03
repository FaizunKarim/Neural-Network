import numpy as np
import os

def load_mnist_from_files(path="data"):
    train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')

    with open(train_labels_path, 'rb') as f:
        train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with open(test_labels_path, 'rb') as f:
        test_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with open(train_images_path, 'rb') as f:
        train_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 28, 28)

    with open(test_images_path, 'rb') as f:
        test_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 28, 28)

    return (train_images, train_labels), (test_images, test_labels)