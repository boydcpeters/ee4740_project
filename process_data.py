import csv
from typing import Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_mnist_data(path: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset from a .csv file and returns the labels and images.

    Parameters
    ----------
    path : str
        Path to the .csv file containing the MNIST dataset.

    Returns
    -------
    labels : (...,) array
        The labels of the images.
    images : (..., 28, 28) array
        The images corresponding to every label.
    """

    print("Data is getting loaded in...")

    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)

        # Count the number of lines, thus images, in the file
        n = sum(1 for row in reader)

        # Pre-initialize the arrays to store the values
        labels = np.zeros(n, dtype=np.uint8)
        images = np.zeros((n, 28, 28), dtype=np.float64)

        # Reset the iterator and recreate the reader object
        csvfile.seek(0)
        reader = csv.reader(csvfile)

        for i, row in tqdm(enumerate(reader)):
            temp = np.array(row, dtype=np.int8)

            # Store the label of the image
            labels[i] = temp[0]

            # Division by 255.0 is for the normalization to get
            # the values in the range [0,1]
            images[i] = np.reshape(temp[1:], (28, 28))

            if normalize:
                images[i] = images[i] / 255.0

    print("Finished data loading.")

    return labels, images
