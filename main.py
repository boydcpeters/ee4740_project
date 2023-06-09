import string
from pathlib import Path

import process_data
from helpers import *

import numpy as np
import matplotlib.pyplot as plt


FULL_COMPARISON_ALL = False  # Recreate figure 5 from report, first generate the data through main_biht.py, main_unnp.py and main_convex.py
FULL_COMPARISON_ALL_ROW = True  # Recreate figure from presentation


if FULL_COMPARISON_ALL:
    path_to_data_raw_convex = f"data\\convex_1500\\metrics_num_m\\"
    path_to_data_processed_biht = f"data\\biht\\metrics_num_m\\processed\\"
    path_to_data_processed_unnp = f"data\\unnp\\metrics_num_m\\processed\\"

    if not Path(path_to_data_processed_biht).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    if not Path(path_to_data_processed_unnp).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    if not Path(path_to_data_raw_convex).exists():
        raise FileNotFoundError("The data does not exist, first generate the data.")

    # Load all the different data arrays
    num_m_convex = process_data.load_arr(path_to_data_raw_convex + "num_m.npy")
    mse_convex = process_data.load_arr(path_to_data_raw_convex + "mse.npy")
    nmse_convex = process_data.load_arr(path_to_data_raw_convex + "nmse.npy")
    psnr_convex = process_data.load_arr(path_to_data_raw_convex + "psnr.npy")

    # Take the mean over all the seeds for every m and every image
    mse_mean_seeds_convex = np.mean(mse_convex, axis=0)
    nmse_mean_seeds_convex = np.mean(nmse_convex, axis=0)
    psnr_mean_seeds_convex = np.mean(psnr_convex, axis=0)

    # Take the mean over all the images for every m
    mse_mean_convex = np.mean(mse_mean_seeds_convex, axis=1)
    mse_std_convex = np.std(mse_mean_seeds_convex, axis=1)

    nmse_mean_convex = np.mean(nmse_mean_seeds_convex, axis=1)
    nmse_std_convex = np.std(nmse_mean_seeds_convex, axis=1)

    psnr_mean_convex = np.mean(psnr_mean_seeds_convex, axis=1)
    psnr_std_convex = np.std(psnr_mean_seeds_convex, axis=1)

    # Load all the different data arrays
    num_m_biht = process_data.load_arr(path_to_data_processed_biht + "num_m.npy")
    mse_mean_biht = process_data.load_arr(path_to_data_processed_biht + "mse_mean.npy")
    mse_std_biht = process_data.load_arr(path_to_data_processed_biht + "mse_std.npy")
    nmse_mean_biht = process_data.load_arr(
        path_to_data_processed_biht + "nmse_mean.npy"
    )
    nmse_std_biht = process_data.load_arr(path_to_data_processed_biht + "nmse_std.npy")
    psnr_mean_biht = process_data.load_arr(
        path_to_data_processed_biht + "psnr_mean.npy"
    )
    psnr_std_biht = process_data.load_arr(path_to_data_processed_biht + "psnr_std.npy")

    # Load all the different data arrays
    num_m_unnp = process_data.load_arr(path_to_data_processed_unnp + "num_m.npy")
    mse_mean_unnp = process_data.load_arr(path_to_data_processed_unnp + "mse_mean.npy")
    mse_std_unnp = process_data.load_arr(path_to_data_processed_unnp + "mse_std.npy")
    nmse_mean_unnp = process_data.load_arr(
        path_to_data_processed_unnp + "nmse_mean.npy"
    )
    nmse_std_unnp = process_data.load_arr(path_to_data_processed_unnp + "nmse_std.npy")
    psnr_mean_unnp = process_data.load_arr(
        path_to_data_processed_unnp + "psnr_mean.npy"
    )
    psnr_std_unnp = process_data.load_arr(path_to_data_processed_unnp + "psnr_std.npy")

    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4, 9))

    axs[0].errorbar(
        num_m_convex,
        mse_mean_convex,
        yerr=mse_std_convex,
        fmt="--o",
        capsize=3,
        label="Convex",
    )
    axs[0].errorbar(
        num_m_biht,
        mse_mean_biht,
        yerr=mse_std_biht,
        fmt="--o",
        capsize=3,
        label="BIHT",
    )
    axs[0].errorbar(
        num_m_unnp,
        mse_mean_unnp,
        yerr=mse_std_unnp,
        fmt="--o",
        capsize=3,
        label="UNNP",
    )

    axs[0].set_xlabel("Number of measurements (m)")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True)

    axs[1].errorbar(
        num_m_convex,
        nmse_mean_convex,
        yerr=nmse_std_convex,
        fmt="--o",
        capsize=3,
        label="Convex",
    )
    axs[1].errorbar(
        num_m_biht,
        nmse_mean_biht,
        yerr=nmse_std_biht,
        fmt="--o",
        capsize=3,
        label="BIHT",
    )
    axs[1].errorbar(
        num_m_unnp,
        nmse_mean_unnp,
        yerr=nmse_std_unnp,
        fmt="--o",
        capsize=3,
        label="UNNP",
    )
    axs[1].set_xlabel("Number of measurements (m)")
    axs[1].set_ylabel("NMSE")
    axs[1].grid(True)

    axs[2].errorbar(
        num_m_convex,
        psnr_mean_convex,
        yerr=psnr_std_convex,
        fmt="--o",
        capsize=3,
        label="Convex",
    )
    axs[2].errorbar(
        num_m_biht,
        psnr_mean_biht,
        yerr=psnr_std_biht,
        fmt="--o",
        capsize=3,
        label="BIHT",
    )
    axs[2].errorbar(
        num_m_unnp,
        psnr_mean_unnp,
        yerr=psnr_std_unnp,
        fmt="--o",
        capsize=3,
        label="UNNP",
    )
    axs[2].set_xlabel("Number of measurements (m)")
    axs[2].set_ylabel("PSNR")
    axs[2].grid(True)

    # fig.tight_layout()

    # Remove duplicate labels, adapted from:
    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.58, 0.95),
        loc="lower center",
        borderaxespad=0,
        ncol=3,
    )
    # fig.legend(title="Number of measurements (m)", loc=7)

    # # fig.tight_layout()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    # Add annotation for subplots
    for i, ax in enumerate(axs):
        ax.text(
            -0.3,
            1.05,
            string.ascii_lowercase[i],
            transform=ax.transAxes,
            size=16,
            weight="bold",
        )

    path_savefig = "data\\figures\\"

    # Create the directory if it does not exist yet
    if not Path(path_savefig).exists():
        Path(path_savefig).mkdir(parents=True)

    if path_savefig is not None:
        fig.savefig(path_savefig + "compare_all.pdf", dpi=200)
        plt.close(fig)

    # plt.show()


if FULL_COMPARISON_ALL_ROW:
    path_to_data_raw_convex = f"data\\convex_1500\\metrics_num_m\\"
    path_to_data_processed_biht = f"data\\biht\\metrics_num_m\\processed\\"
    path_to_data_processed_unnp = f"data\\unnp\\metrics_num_m\\processed\\"

    if not Path(path_to_data_processed_biht).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    if not Path(path_to_data_processed_unnp).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    if not Path(path_to_data_raw_convex).exists():
        raise FileNotFoundError("The data does not exist, first generate the data.")

    # Load all the different data arrays
    num_m_convex = process_data.load_arr(path_to_data_raw_convex + "num_m.npy")
    mse_convex = process_data.load_arr(path_to_data_raw_convex + "mse.npy")
    nmse_convex = process_data.load_arr(path_to_data_raw_convex + "nmse.npy")
    psnr_convex = process_data.load_arr(path_to_data_raw_convex + "psnr.npy")

    # Take the mean over all the seeds for every m and every image
    mse_mean_seeds_convex = np.mean(mse_convex, axis=0)
    nmse_mean_seeds_convex = np.mean(nmse_convex, axis=0)
    psnr_mean_seeds_convex = np.mean(psnr_convex, axis=0)

    # Take the mean over all the images for every m
    mse_mean_convex = np.mean(mse_mean_seeds_convex, axis=1)
    mse_std_convex = np.std(mse_mean_seeds_convex, axis=1)

    nmse_mean_convex = np.mean(nmse_mean_seeds_convex, axis=1)
    nmse_std_convex = np.std(nmse_mean_seeds_convex, axis=1)

    psnr_mean_convex = np.mean(psnr_mean_seeds_convex, axis=1)
    psnr_std_convex = np.std(psnr_mean_seeds_convex, axis=1)

    # Load all the different data arrays
    num_m_biht = process_data.load_arr(path_to_data_processed_biht + "num_m.npy")
    mse_mean_biht = process_data.load_arr(path_to_data_processed_biht + "mse_mean.npy")
    mse_std_biht = process_data.load_arr(path_to_data_processed_biht + "mse_std.npy")
    nmse_mean_biht = process_data.load_arr(
        path_to_data_processed_biht + "nmse_mean.npy"
    )
    nmse_std_biht = process_data.load_arr(path_to_data_processed_biht + "nmse_std.npy")
    psnr_mean_biht = process_data.load_arr(
        path_to_data_processed_biht + "psnr_mean.npy"
    )
    psnr_std_biht = process_data.load_arr(path_to_data_processed_biht + "psnr_std.npy")

    # Load all the different data arrays
    num_m_unnp = process_data.load_arr(path_to_data_processed_unnp + "num_m.npy")
    mse_mean_unnp = process_data.load_arr(path_to_data_processed_unnp + "mse_mean.npy")
    mse_std_unnp = process_data.load_arr(path_to_data_processed_unnp + "mse_std.npy")
    nmse_mean_unnp = process_data.load_arr(
        path_to_data_processed_unnp + "nmse_mean.npy"
    )
    nmse_std_unnp = process_data.load_arr(path_to_data_processed_unnp + "nmse_std.npy")
    psnr_mean_unnp = process_data.load_arr(
        path_to_data_processed_unnp + "psnr_mean.npy"
    )
    psnr_std_unnp = process_data.load_arr(path_to_data_processed_unnp + "psnr_std.npy")

    # Create the figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axs[0].errorbar(
        num_m_convex,
        mse_mean_convex,
        yerr=mse_std_convex,
        fmt="--o",
        capsize=3,
        label="Convex",
    )
    axs[0].errorbar(
        num_m_biht,
        mse_mean_biht,
        yerr=mse_std_biht,
        fmt="--o",
        capsize=3,
        label="BIHT",
    )
    axs[0].errorbar(
        num_m_unnp,
        mse_mean_unnp,
        yerr=mse_std_unnp,
        fmt="--o",
        capsize=3,
        label="UNNP",
    )

    axs[0].set_xlabel("Number of measurements (m)")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True)

    axs[1].errorbar(
        num_m_convex,
        nmse_mean_convex,
        yerr=nmse_std_convex,
        fmt="--o",
        capsize=3,
        label="Convex",
    )
    axs[1].errorbar(
        num_m_biht,
        nmse_mean_biht,
        yerr=nmse_std_biht,
        fmt="--o",
        capsize=3,
        label="BIHT",
    )
    axs[1].errorbar(
        num_m_unnp,
        nmse_mean_unnp,
        yerr=nmse_std_unnp,
        fmt="--o",
        capsize=3,
        label="UNNP",
    )
    axs[1].set_xlabel("Number of measurements (m)")
    axs[1].set_ylabel("NMSE")
    axs[1].grid(True)

    axs[2].errorbar(
        num_m_convex,
        psnr_mean_convex,
        yerr=psnr_std_convex,
        fmt="--o",
        capsize=3,
        label="Convex",
    )
    axs[2].errorbar(
        num_m_biht,
        psnr_mean_biht,
        yerr=psnr_std_biht,
        fmt="--o",
        capsize=3,
        label="BIHT",
    )
    axs[2].errorbar(
        num_m_unnp,
        psnr_mean_unnp,
        yerr=psnr_std_unnp,
        fmt="--o",
        capsize=3,
        label="UNNP",
    )
    axs[2].set_xlabel("Number of measurements (m)")
    axs[2].set_ylabel("PSNR")
    axs[2].grid(True)

    # fig.tight_layout()

    # Remove duplicate labels, adapted from:
    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5225, 0.91),
        loc="lower center",
        borderaxespad=0,
        ncol=3,
    )
    # fig.legend(title="Number of measurements (m)", loc=7)

    # # fig.tight_layout()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.85])

    # Add annotation for subplots
    for i, ax in enumerate(axs):
        ax.text(
            -0.15,
            1.05,
            string.ascii_lowercase[i],
            transform=ax.transAxes,
            size=16,
            weight="bold",
        )

    path_savefig = "data\\figures\\"

    # Create the directory if it does not exist yet
    if not Path(path_savefig).exists():
        Path(path_savefig).mkdir(parents=True)

    if path_savefig is not None:
        fig.savefig(path_savefig + "compare_all_row.pdf", dpi=200)
        fig.savefig(path_savefig + "compare_all_row.png", dpi=200)
        plt.close(fig)

    # plt.show()
