import numpy as np


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
        _description_
    """
    mse = compute_mse(x_im, y_im)

    if mse == 0:
        # MSE is zero means that there is no noise in the signal
        # so PSNR does not have any importance
        return 100

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr
