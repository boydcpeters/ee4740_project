from typing import Tuple
from pathlib import Path

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
    "data\\raw\\mnist_test.csv", normalize=True, max_rows=200
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
            # TODO: test the psnr calculation
            psnr[i] = compute_psnr(x_im, x_hat_norm)

        return s_levels, mse, nmse, psnr

    SEED = 1
    S_LEVEL_MAX = 784
    NUM_IMAGES = 300

    # Number of measurements
    m = 500

    A = cs_func.create_A(m, 784, seed=SEED)
    m, n = A.shape

    s_value = min(m, S_LEVEL_MAX)
    s_levels = np.arange(1, s_value + 1)

    s_levels = np.tile(s_levels, (s_value, 1))
    mse = np.zeros(s_levels.shape, dtype=np.float64)
    nmse = np.zeros(s_levels.shape, dtype=np.float64)
    psnr = np.zeros(s_levels.shape, dtype=np.float64)

    # Randomly select images from the dataset
    idx_possible = np.arange(0, images.shape[0])
    rng = np.random.default_rng(seed=SEED)
    idx = rng.choice(idx_possible, size=(NUM_IMAGES), replace=False)

    labels_metrics = np.zeros(idx.shape[0], dtype=np.uint8)
    images_metrics = np.zeros((idx.shape[0], 28, 28), dtype=np.float64)

    for i in tqdm(range(idx.shape[0])):
        # Store the used images in an array
        labels_metrics[i] = labels[idx[i]]
        images_metrics[i] = images[idx[i]]

        # Load an image
        x_im = images[idx[i]]

        # Calculate the performance metrics
        _, mse[i, :], nmse[i, :], psnr[i, :] = metrics_s_levels_biht(A, x_im)

    # Save the data
    path_to_save = f"data\\biht\\A{m}_seed{SEED}\\"

    # Create the directory if it does not exist yet
    if not Path(path_to_save).exists():
        Path(path_to_save).mkdir(parents=True)

    # Save the A matrix
    process_data.save_arr(path_to_save + f"A{m}.npy", A)
    process_data.save_arr(path_to_save + f"labels.npy", labels_metrics)
    process_data.save_arr(path_to_save + f"images.npy", images_metrics)
    process_data.save_arr(path_to_save + f"s_levels.npy", s_levels)
    process_data.save_arr(path_to_save + f"mse.npy", mse)
    process_data.save_arr(path_to_save + f"nmse.npy", nmse)
    process_data.save_arr(path_to_save + f"psnr.npy", psnr)

    fig3, axs3 = visualize.plot_metrics(
        s_levels,
        (mse, nmse, psnr),
        "Sparsity level (s)",
        ("MSE", "NMSE", "PSNR"),
        ci_flag=True,
    )

    plt.show()
