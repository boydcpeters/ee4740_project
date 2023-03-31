import process_data
import visualize
import cs_func
import models
from helpers import *

import numpy as np
import matplotlib.pyplot as plt

SPARSITY_DISTRIBUTION_FLAG = True


labels, images = process_data.load_mnist_data(
    "data\mnist_train.csv", normalize=True, max_rows=None
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
