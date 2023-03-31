import process_data
import visualize
import cs_func
import models
from helpers import *

import numpy as np
import matplotlib.pyplot as plt

SPARSITY_DISTRIBUTION_FLAG = False
BIHT_RUN_FLAG = True


labels, images = process_data.load_mnist_data(
    "data\mnist_test.csv", normalize=True, max_rows=None
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
