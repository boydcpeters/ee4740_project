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
    "data\mnist_test.csv", normalize=True, max_rows=10
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
    seed = 1

    S_LEVEL_MAX = 784

    # Number of measurements
    m = 400

    A = cs_func.create_A(m, 784, seed=seed)
    m, n = A.shape

    # Load an image
    x_im = images[0]
    x = x_im.flatten()

    y = cs_func.calc_y(A, x)

    mse = np.zeros(m)
    nmse = np.zeros(m)

    for i in tqdm(range(min(m, S_LEVEL_MAX))):
        x_hat = models.biht(A, y, (i + 1), max_iter=100, mode="l1", verbose=False)
        x_hat = np.reshape(x_hat, (28, 28))

        mse[i] = compute_mse(x_im, x_hat)
        nmse[i] = compute_nmse(x_im, x_hat)

    s_levels = np.arange(1, m + 1)

    fig3, axs3 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axs3[0].plot(s_levels, mse)
    axs3[1].plot(s_levels, nmse)

    axs3[0].set_xlabel("Sparsity level (s)")
    axs3[0].set_ylabel("MSE")

    axs3[1].set_xlabel("Sparsity level (s)")
    axs3[1].set_ylabel("NMSE")

    fig3.tight_layout()
    plt.show()
