from typing import Tuple

import numpy as np

SEED = 1


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Function normalizes the data in x to the range [0, 1].

    Parameters
    ----------
    x : np.ndarray
        Data

    Returns
    -------
    np.ndarray
        Normalized data
    """
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))


def compute_mse(x_im: np.ndarray, y_im: np.ndarray) -> float:
    """
    Function computes the Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    x_im : np.ndarray
        Image represented by a np.ndarray.
    y_im : np.ndarray
        Image represented by a np.ndarray.

    Returns
    -------
    float
        Mean Squared Error (MSE) value
    """

    return np.mean((x_im - y_im) ** 2)


def compute_nmse(x_im: np.ndarray, y_im: np.ndarray) -> float:
    """
    Function computes the Normalized Mean Squared Error (NMSE) between two images.

    Parameters
    ----------
    x_im : np.ndarray
        Image represented by a np.ndarray.
    y_im : np.ndarray
        Image represented by a np.ndarray.

    Returns
    -------
    float
        Normalized Mean Squared Error (NMSE) value
    """

    return np.mean(((x_im / np.linalg.norm(x_im)) - (y_im / np.linalg.norm(y_im))) ** 2)


def compute_psnr(x_im: np.ndarray, y_im: np.ndarray, max_pixel: float = 1.0) -> float:
    """
    Function computes the Peak Signal-to-Noise Ratio (PSNR) value between two images.

    Parameters
    ----------
    x_im : np.ndarray
        Image represented by a np.ndarray.
    y_im : np.ndarray
        Image represented by a np.ndarray.
    max_pixel : float, optional
        Maximum pixel value specified by the user, by default 1.0

    Returns
    -------
    psnr : float
        Peak Signal-to-Noise Ratio (PSNR) value
    """
    mse = compute_mse(x_im, y_im)

    if mse == 0:
        # MSE is zero means that there is no noise in the signal
        # so PSNR does not have any importance
        return 100

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def retrieve_s_levels_images(images: np.ndarray) -> np.ndarray:
    """
    Function returns for every image the sparsity level (number of nonzero values).

    Parameters
    ----------
    images : np.ndarray
        Array with all the images. (num of images, h, w)

    Returns
    -------
    np.ndarray
        Array with the sparsity level of every image.
    """
    # Determine the minimum value that is not zero
    min_value = np.amin(images[np.nonzero(images)])

    # Flag all the values larger or equal to the minimum value
    images_nonzeros = images >= min_value

    # For every image store the number of nonzeros
    s_images = np.zeros(images.shape[0])
    for i in range(s_images.shape[0]):
        s_images[i] = np.count_nonzero(images_nonzeros[i, :, :])

    return s_images


def get_seeds(seed: int = SEED):
    # Create the random number generator
    rng = np.random.default_rng(seed)

    # Get the seeds
    seeds = rng.integers(1, np.iinfo(np.int64).max, size=(5,))

    return seeds


def get_idx_row_images(seed: int = SEED):
    # Create the random number generator
    rng = np.random.default_rng(seed)

    # Get the indices of the rows for the images
    idx_row = rng.integers(0, 10000, size=(20,))

    return idx_row
