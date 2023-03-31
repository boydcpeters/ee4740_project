import csv
from typing import Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_mnist_data(
    path: str, normalize: bool = True, max_rows: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset from a .csv file and returns the labels and images.

    Parameters
    ----------
    path : str
        Path to the .csv file containing the MNIST dataset.
    normalize : bool
        Indicates whether the data should be normalized.
    max_rows : int
        Maximum number of rows that should be read in, by default None. In this case
        all the rows will be read.

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

        if max_rows is not None:
            n = min(n, max_rows)

        # Pre-initialize the arrays to store the values
        labels = np.zeros(n, dtype=np.uint8)
        images = np.zeros((n, 28, 28), dtype=np.float64)

        # Reset the iterator and recreate the reader object
        csvfile.seek(0)
        reader = csv.reader(csvfile)

        with tqdm(total=n) as pbar:
            for i, row in enumerate(reader):
                temp = np.array(row, dtype=np.uint8)

                # Store the label of the image
                labels[i] = temp[0]

                # Division by 255.0 is for the normalization to get
                # the values in the range [0,1]
                images[i] = np.reshape(temp[1:], (28, 28))

                if normalize:
                    images[i] = images[i] / 255.0

                # Update the progress bar
                pbar.update(1)

                if max_rows is not None:
                    # Stop iterating if maximum number of rows is reached
                    if i == max_rows - 1:
                        break

    print("Finished data loading.")

    return labels, images


def save_arr(path: str, x: np.ndarray):
    """
    Function saves the x array to an .npy file.

    Parameters
    ----------
    path : str
        Location to store x array.
    x : np.ndarray
        The array to store.
    """

    print(f"Saving the array to: {path}")
    np.save(path, x)
    print(f"Finished saving.")


def load_arr(path: str) -> np.ndarray:
    """
    Function loads the x array from an .npy file.

    Parameters
    ----------
    path : str
        Location where x array is stored.

    Returns
    -------
    np.ndarray
        The loaded array.
    """
    print(f"Loading the array from: {path}")
    x = np.load(path)
    print(f"Finished loading.")

    return x
