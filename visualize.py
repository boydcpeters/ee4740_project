import numpy as np
import matplotlib.pyplot as plt


def plot_image(image: np.ndarray, title: str = None):
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)

    ax1.imshow(image, cmap="viridis")

    if title is not None:
        ax1.set_title(title)

    return fig1, ax1


def plot_images(images: np.ndarray):
    n = images.shape[0]

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows)

    i_row = 0
    i_col = 0
    for i in range(n):
        axs[i_row, i_col].imshow(images[i], cmap="viridis")

        i_col += 1

        if i_col == ncols:
            i_col = 0
            i_row += 1

    fig.tight_layout()

    return fig, axs
