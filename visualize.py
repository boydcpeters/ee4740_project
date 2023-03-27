from typing import Sequence, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image: np.ndarray, title: str = None, cmap: str = "viridis"):
    """
    Function creates a figure and axes object with the image plotted.

    Parameters
    ----------
    image : np.ndarray
        Array with pixel values to display.
    title : str, optional
        The title of the figure, by default None.
    cmap : str, optional
        Colormap used for the visualization of the image, by default "viridis".

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        The top level container for all the plot elements.

    ax : :py:class:`matplotlib.axes.Axes`
        A single :py:class:`matplotlib.axes.Axes` object.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.imshow(image, cmap=cmap)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_images(
    images: Union[np.ndarray, Sequence[np.ndarray]],
    title: str = None,
    cmap: str = "viridis",
):
    """
    Function creates a figure and axes object with the images plotted.

    Parameters
    ----------
    images : np.ndarray | Sequence[np.ndarray]
        Arrays with pixel values to display. The first axis denotes a new image.
        Alternatively, a Sequence of arrays can be provided.
    title : str, optional
        The title of the figure, by default None.
    cmap : str, optional
        Colormap used for the visualization of the image, by default "viridis".

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        The top level container for all the plot elements.

    axs : :py:class:`matplotlib.axes.Axes`
        A sequence of :py:class:`matplotlib.axes.Axes` objects.
    """

    if isinstance(images, np.ndarray):
        n = images.shape[0]
    elif isinstance(images, Sequence):
        n = len(images)

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows)

    if len(axs.shape) != 2:
        axs = axs[np.newaxis, ...]

    i_row = 0
    i_col = 0
    for i in range(n):
        axs[i_row, i_col].imshow(images[i], cmap="viridis")

        i_col += 1

        if i_col == ncols:
            i_col = 0
            i_row += 1

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    return fig, axs
