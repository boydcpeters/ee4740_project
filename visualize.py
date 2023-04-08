from typing import Sequence, Union, List, Tuple
import string
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    titles: Union[List[str], Tuple[str]] = None,
    suptitle: str = None,
    cmap: str = "viridis",
    figsize: Tuple[Union[int, float]] = None,
    add_cbar: bool = True,
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

    if n <= 3:
        ncols = n
        nrows = 1
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = [6.4, 4.8]  # just default values

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=figsize)

    if len(axs.shape) != 2:
        axs = axs[np.newaxis, ...]

    i_row = 0
    i_col = 0
    for i in range(n):
        im = axs[i_row, i_col].imshow(images[i], cmap=cmap)

        # Next part for the colorbar is inspired by:
        # https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
        if add_cbar:
            divider = make_axes_locatable(axs[i_row, i_col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        axs[i_row, i_col].text(
            -0.15,
            1.1,
            string.ascii_lowercase[i],
            transform=axs[i_row, i_col].transAxes,
            size=16,
            weight="bold",
        )

        if titles is not None:
            axs[i_row, i_col].set_title(titles[i])

        i_col += 1

        if i_col == ncols:
            i_col = 0
            i_row += 1

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    # # Add annotation for subplots
    # for i, ax in enumerate(axs):
    #     ax.text(
    #         -0.15,
    #         1.05,
    #         string.ascii_lowercase[i],
    #         transform=ax.transAxes,
    #         size=16,
    #         weight="bold",
    #     )

    return fig, axs


def plot_metrics(
    x,
    metrics,
    xlabel: Union[str, Sequence[str]] = None,
    ylabel: Union[str, Sequence[str]] = None,
    ci_flag: bool = True,
    ci_label: str = "95% confidence interval",
    z: float = 1.96,
):
    fig, axs = plt.subplots(nrows=1, ncols=3)

    for i, metric in enumerate(metrics):
        metric_mean = np.mean(metric, axis=0)

        axs[i].plot(x[0, :], metric_mean, label="Mean")

        if ci_flag:
            metric_std = np.std(metric, axis=0)
            metric_ci = z * metric_std / np.sqrt(metric.shape[0])

            axs[i].fill_between(
                x[0, :],
                (metric_mean - metric_ci),
                (metric_mean + metric_ci),
                color="b",
                alpha=0.1,
                label=ci_label,
            )

        if isinstance(xlabel, tuple) or isinstance(xlabel, list):
            xlabel_temp = xlabel[i]
        else:
            xlabel_temp = xlabel

        if isinstance(ylabel, tuple) or isinstance(ylabel, list):
            ylabel_temp = ylabel[i]
        else:
            ylabel_temp = ylabel

        axs[i].set_xlabel(xlabel_temp)
        axs[i].set_ylabel(ylabel_temp)

    fig.tight_layout()

    return fig, axs
