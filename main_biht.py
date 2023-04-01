from typing import Tuple

import process_data
import visualize
import cs_func
import models
from helpers import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SPARSITY_DISTRIBUTION_FLAG = False
BIHT_RUN_FLAG = False
BIHT_CHECK_S_LEVELS_FLAG = True


labels, images = process_data.load_mnist_data(
    "data\mnist_test.csv", normalize=True, max_rows=200
)

if SPARSITY_DISTRIBUTION_FLAG:
    s_images = retrieve_s_levels_images(images)

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)

    unique, count = np.unique(s_images, return_counts=True)

    # bins = np.arange(np.amin(s_images), np.amax(s_images) + 1) - 0.5

    s_mean = np.mean(s_images)
    s_std = np.std(s_images)
    print(f"Mean sparsity level: {s_mean}, standard deviation: {s_std}")

    count_norm = count / np.sum(count)

    ax1.bar(unique, count_norm, width=1)
    ax1.set_xlabel("Sparsity level (s)")
    ax1.set_ylabel("Density of occurences")

    plt.show()


if BIHT_RUN_FLAG:
    # Load an image
    x_im = images[0]
    x = x_im.flatten()

    # Create the measurement matrix and calculate y
    A = cs_func.create_A(800, 784)
    y = cs_func.calc_y(A, x)

    # Estimate x based on y and A with BIHT
    # TODO: fix l2-mode, it is currently not working
    x_hat = models.biht(A, y, 150, max_iter=300, mode="l1", verbose=True)
    x_hat = np.reshape(x_hat, (28, 28))

    print(f"MSE: {compute_mse(x_im, x_hat)}")
    print(f"NMSE: {compute_nmse(x_im, x_hat)}")

    fig2, axs2 = visualize.plot_images((x_im, x_hat))
    plt.show()


if BIHT_CHECK_S_LEVELS_FLAG:

    def metrics_s_levels_biht(
        A: np.ndarray, im: np.ndarray, s_level_max: int = 784, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = A.shape

        x = im.flatten()
        y = cs_func.calc_y(A, x)

        s_value = min(m, s_level_max)
        s_levels = np.arange(1, s_value + 1)

        mse = np.zeros(s_levels.shape, dtype=np.float64)
        nmse = np.zeros(s_levels.shape, dtype=np.float64)
        psnr = np.zeros(s_levels.shape, dtype=np.float64)

        for i in tqdm(range(s_value)) if verbose else range(s_value):
            x_hat = models.biht(A, y, (i + 1), max_iter=100, mode="l1", verbose=False)
            x_hat = np.reshape(x_hat, (28, 28))

            mse[i] = compute_mse(x_im, x_hat)
            nmse[i] = compute_nmse(x_im, x_hat)

            x_hat_norm = normalize(x_hat)
            psnr[i] = compute_psnr(x_im, x_hat_norm)

        return s_levels, mse, nmse, psnr

    S_LEVEL_MAX = 784

    seed = 1

    # Number of measurements
    m = 400

    A = cs_func.create_A(m, 784, seed=seed)
    m, n = A.shape

    s_value = min(m, S_LEVEL_MAX)
    s_levels = np.arange(1, s_value + 1)

    s_levels = np.tile(s_levels, (s_value, 1))
    mse = np.zeros(s_levels.shape, dtype=np.float64)
    nmse = np.zeros(s_levels.shape, dtype=np.float64)
    psnr = np.zeros(s_levels.shape, dtype=np.float64)

    for i in tqdm(range(100)):
        # Load an image
        x_im = images[i]

        _, mse[i, :], nmse[i, :], psnr[i, :] = metrics_s_levels_biht(A, x_im)

    mse_mean = np.mean(mse, axis=0)
    mse_std = np.std(mse, axis=0)

    nmse_mean = np.mean(nmse, axis=0)
    nmse_std = np.std(nmse, axis=0)

    psnr_mean = np.mean(psnr, axis=0)
    psnr_std = np.std(psnr, axis=0)

    fig3, axs3 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axs3[0].plot(s_levels[0, :], mse_mean)
    axs3[1].plot(s_levels[0, :], nmse_mean)
    axs3[2].plot(s_levels[0, :], psnr_mean)

    axs3[0].set_xlabel("Sparsity level (s)")
    axs3[0].set_ylabel("MSE")

    axs3[1].set_xlabel("Sparsity level (s)")
    axs3[1].set_ylabel("NMSE")

    axs3[2].set_xlabel("Sparsity level (s)")
    axs3[2].set_ylabel("PSNR")

    fig3.tight_layout()
    plt.show()
