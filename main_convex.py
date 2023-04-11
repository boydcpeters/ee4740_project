from typing import Tuple
from pathlib import Path

import process_data
import visualize
import cs_func
import models
import helpers

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

CONVEX_RUN_M = True
PLOT_RESULTS_CONVEX_TEST_NUM_M = True

if CONVEX_RUN_M:
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_test.csv", normalize=True, max_rows=None
    )

    seeds = helpers.get_seeds()
    idx_row_images = helpers.get_idx_row_images()

    num_m = np.array([25, 100, 200, 500, 1000, 1500])

    mse = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0]), dtype=np.float64
    )
    nmse = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0]), dtype=np.float64
    )
    psnr = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0]), dtype=np.float64
    )

    for i in tqdm(range(seeds.shape[0])):  # Loop over all the seeds
        seed = seeds[i]

        # Loop over all the different number of measurments
        for j in range(num_m.shape[0]):
            m = num_m[j]

            A = cs_func.create_A(m, 784, seed=seed)

            # Loop over all the images
            for k in range(idx_row_images.shape[0]):
                # Get the required images
                idx = idx_row_images[k]
                x_im = images[idx]
                x = x_im.flatten()

                # Calculate y = np.sign(A @ x)
                y = cs_func.calc_y(A, x)

                # Reconstruct x with BIHT algorithm
                x_hat = models.convex(A, y)
                x_hat = np.reshape(x_hat, (28, 28))

                # Compute the metrics
                mse[i, j, k] = helpers.compute_mse(x_im, x_hat)
                nmse[i, j, k] = helpers.compute_nmse(x_im, x_hat)

                x_hat_norm = helpers.normalize(x_hat)
                psnr[i, j, k] = helpers.compute_psnr(x_im, x_hat_norm)

    path_to_data = f"data\\convex_1500\\metrics_num_m1\\"

    # If the path does not yet exists, create it
    if not Path(path_to_data).exists():
        Path(path_to_data).mkdir(parents=True)

    # Save all the different data arrays
    process_data.save_arr(path_to_data + "num_m.npy", num_m)
    process_data.save_arr(path_to_data + "mse.npy", mse)
    process_data.save_arr(path_to_data + "nmse.npy", nmse)
    process_data.save_arr(path_to_data + "psnr.npy", psnr)

if PLOT_RESULTS_CONVEX_TEST_NUM_M:
    path_to_data = f"data\\convex_1500\\metrics_num_m1\\"

    if not Path(path_to_data).exists():
        raise FileNotFoundError("The data does not exist, first generate the data.")

    # Load all the different data arrays
    num_m = process_data.load_arr(path_to_data + "num_m.npy")
    mse = process_data.load_arr(path_to_data + "mse.npy")
    nmse = process_data.load_arr(path_to_data + "nmse.npy")
    psnr = process_data.load_arr(path_to_data + "psnr.npy")

    # Take the mean over all the seeds for every m and every image
    mse_mean_seeds = np.mean(mse, axis=0)
    nmse_mean_seeds = np.mean(nmse, axis=0)
    psnr_mean_seeds = np.mean(psnr, axis=0)

    # Take the mean over all the images for every m
    mse_mean = np.mean(mse_mean_seeds, axis=1)
    mse_std = np.std(mse_mean_seeds, axis=1)

    nmse_mean = np.mean(nmse_mean_seeds, axis=1)
    nmse_std = np.std(nmse_mean_seeds, axis=1)

    psnr_mean = np.mean(psnr_mean_seeds, axis=1)
    psnr_std = np.std(psnr_mean_seeds, axis=1)

    # Create the figure
    fig4, axs4 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axs4[0].errorbar(num_m, mse_mean, yerr=mse_std, fmt="--o", capsize=3)
    axs4[0].set_xlabel("Number of measurements (m)")
    axs4[0].set_ylabel("MSE")
    axs4[0].grid(True)

    axs4[1].errorbar(num_m, nmse_mean, yerr=nmse_std, fmt="--o", capsize=3)
    axs4[1].set_xlabel("Number of measurements (m)")
    axs4[1].set_ylabel("NMSE")
    axs4[1].grid(True)

    axs4[2].errorbar(num_m, psnr_mean, yerr=psnr_std, fmt="--o", capsize=3)
    axs4[2].set_xlabel("Number of measurements (m)")
    axs4[2].set_ylabel("PSNR")
    axs4[2].grid(True)

    fig4.tight_layout()

    plt.show()
