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

SPARSITY_DISTRIBUTION_FLAG = False
BIHT_RUN_FLAG = False
BIHT_CHECK_S_LEVELS_FLAG = False
BIHT_TEST_NUM_M = False
PROCESS_DATA_BIHT_TEST_NUM_M = True
PLOT_RESULTS_BIHT_TEST_NUM_M = True


if SPARSITY_DISTRIBUTION_FLAG:
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_train.csv", normalize=True, max_rows=None
    )

    s_images = helpers.retrieve_s_levels_images(images)

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
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_train.csv", normalize=True, max_rows=10
    )

    # Load an image
    x_im = images[0]
    x = x_im.flatten()

    # Create the measurement matrix and calculate y
    A = cs_func.create_A(800, 784)
    y = cs_func.calc_y(A, x)

    # Estimate x based on y and A with BIHT
    x_hat = models.biht(A, y, 150, max_iter=300, mode="l1", verbose=True)
    x_hat = np.reshape(x_hat, (28, 28))

    print(f"MSE: {helpers.compute_mse(x_im, x_hat)}")
    print(f"NMSE: {helpers.compute_nmse(x_im, x_hat)}")

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

            mse[i] = helpers.compute_mse(x_im, x_hat)
            nmse[i] = helpers.compute_nmse(x_im, x_hat)

            x_hat_norm = helpers.normalize(x_hat)
            # TODO: test the psnr calculation
            psnr[i] = helpers.compute_psnr(x_im, x_hat_norm)

        return s_levels, mse, nmse, psnr

    SEED = 1
    S_LEVEL_MAX = 784
    NUM_IMAGES = 60000

    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_train.csv", normalize=True, max_rows=None
    )

    # Number of measurements
    m = 25

    A = cs_func.create_A(m, 784, seed=SEED)
    m, n = A.shape

    s_value = min(m, S_LEVEL_MAX)
    s_levels = np.arange(1, s_value + 1)

    print(f"s_levels.shape: {s_levels.shape}")

    s_levels = np.tile(s_levels, (NUM_IMAGES, 1))
    mse = np.zeros((NUM_IMAGES, s_levels.shape[1]), dtype=np.float64)
    nmse = np.zeros((NUM_IMAGES, s_levels.shape[1]), dtype=np.float64)
    psnr = np.zeros((NUM_IMAGES, s_levels.shape[1]), dtype=np.float64)

    # Randomly select images from the dataset
    idx_possible = np.arange(0, images.shape[0])
    rng = np.random.default_rng(seed=SEED)
    idx = rng.choice(idx_possible, size=(NUM_IMAGES), replace=False)

    labels_metrics = np.zeros(idx.shape[0], dtype=np.uint8)
    images_rows_metrics = np.zeros((idx.shape[0], 28, 28), dtype=np.float64)

    for i in tqdm(range(idx.shape[0])):
        # Store the used images in an array
        labels_metrics[i] = labels[idx[i]]
        images_rows_metrics[i] = idx[i]

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
    process_data.save_arr(path_to_save + f"images_rows.npy", images_rows_metrics)
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


if BIHT_TEST_NUM_M:
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
                x_hat = models.biht(A, y, 200, max_iter=100, mode="l1", verbose=False)
                x_hat = np.reshape(x_hat, (28, 28))

                # Compute the metrics
                mse[i, j, k] = helpers.compute_mse(x_im, x_hat)
                nmse[i, j, k] = helpers.compute_nmse(x_im, x_hat)

                x_hat_norm = helpers.normalize(x_hat)
                psnr[i, j, k] = helpers.compute_psnr(x_im, x_hat_norm)

    path_to_data = f"data\\biht\\metrics_num_m\\raw\\"

    # If the path does not yet exists, create it
    if not Path(path_to_data).exists():
        Path(path_to_data).mkdir(parents=True)

    # Save all the different data arrays
    process_data.save_arr(path_to_data + "num_m.npy", num_m)
    process_data.save_arr(path_to_data + "mse.npy", mse)
    process_data.save_arr(path_to_data + "nmse.npy", nmse)
    process_data.save_arr(path_to_data + "psnr.npy", psnr)


if PROCESS_DATA_BIHT_TEST_NUM_M:
    path_to_data_raw = f"data\\biht\\metrics_num_m\\raw\\"
    if not Path(path_to_data_raw).exists():
        raise FileNotFoundError("The data does not exist, first generate the data.")

    path_to_data_processed = f"data\\biht\\metrics_num_m\\processed\\"
    if not Path(path_to_data_processed).exists():
        Path(path_to_data_processed).mkdir(parents=True)

    # Load all the different data arrays
    num_m = process_data.load_arr(path_to_data_raw + "num_m.npy")
    mse = process_data.load_arr(path_to_data_raw + "mse.npy")
    nmse = process_data.load_arr(path_to_data_raw + "nmse.npy")
    psnr = process_data.load_arr(path_to_data_raw + "psnr.npy")

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

    # Save all the different data arrays
    process_data.save_arr(path_to_data_processed + "num_m.npy", num_m)
    process_data.save_arr(path_to_data_processed + "mse_mean.npy", mse_mean)
    process_data.save_arr(path_to_data_processed + "mse_std.npy", mse_std)
    process_data.save_arr(path_to_data_processed + "nmse_mean.npy", nmse_mean)
    process_data.save_arr(path_to_data_processed + "nmse_std.npy", nmse_std)
    process_data.save_arr(path_to_data_processed + "psnr_mean.npy", psnr_mean)
    process_data.save_arr(path_to_data_processed + "psnr_std.npy", psnr_std)


if PLOT_RESULTS_BIHT_TEST_NUM_M:
    path_to_data_processed = f"data\\biht\\metrics_num_m\\processed\\"

    if not Path(path_to_data_processed).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    # Load all the different data arrays
    num_m = process_data.load_arr(path_to_data_processed + "num_m.npy")
    mse_mean = process_data.load_arr(path_to_data_processed + "mse_mean.npy")
    mse_std = process_data.load_arr(path_to_data_processed + "mse_std.npy")
    nmse_mean = process_data.load_arr(path_to_data_processed + "nmse_mean.npy")
    nmse_std = process_data.load_arr(path_to_data_processed + "nmse_std.npy")
    psnr_mean = process_data.load_arr(path_to_data_processed + "psnr_mean.npy")
    psnr_std = process_data.load_arr(path_to_data_processed + "psnr_std.npy")

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
