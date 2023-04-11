import time
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import models
import cs_func
import process_data
import visualize
import helpers

# NOTE: this code requires CUDA to run

# First determine whether the model should be run on the CPU or GPU,
# CPU seems to be quicker (since the model is relatively small, so transferring it to the
# GPU does not seem worth it.)
GPU_FLAG = False
dtype, device = models.get_dtype_device(GPU_FLAG)

TEST_RUN_UNNP = False  # Example run with one image for UNNP
COMPARE_LOSS_DENOM_SQUARE = False  # Recreate figure 3 in report
UNNP_TEST_NUM_M = False  # Run this to perform the simulations to get UNNP reconstruction performance for different number of measurements
UNNP_TEST_NUM_M_LEAKYRELU = False  # Run this to perform the simulations to get UNNP (LeakyReLU) reconstruction performance for different number of measurements

# Process data
PROCESS_DATA_UNNP_TEST_NUM_M = False
PROCESS_DATA_UNNP_TEST_NUM_M_LEAKYRELU = False

# Plot results
PLOT_RESULTS_UNNP_TEST_NUM_M = False
PLOT_RESULTS_UNNP_TEST_NUM_M_LEAKYRELU = False
PLOT_RESULTS_UNNP_TEST_NUM_M_LEAKYRELU_OLNY_NMSE = False  #  Recreate figure 4 in report
PLOT_RESULTS_COMPARE_UNNP_TEST_NUM_M_RELU_VS_LEAKYRELU = (
    True  # Plot all the metrics for comparison UNNP ReLU vs LeakyReLU
)

if TEST_RUN_UNNP:
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_test.csv", normalize=True, max_rows=10
    )

    x_im = images[0]

    A = cs_func.create_A(1000, 784, seed=1)
    y = cs_func.calc_y(A, x_im.flatten())

    # Move the data over to the set device
    A_var = torch.from_numpy(A).float().to(device)
    y_var = torch.from_numpy(y).float().to(device)

    # Set the parameters for the decoder network
    num_channels = [25, 15, 10]
    output_depth = 1  # number of output channels

    # Create the network
    net = models.DecoderNet(
        num_output_channels=output_depth,
        num_channels_up=num_channels,
        leakyrelu_flag=False,
    ).type(dtype)

    # Move the net to the correct device
    net.to(device)

    # Store the time before the fitting
    t0 = time.time()

    # Run the fitting
    loss_t, net_input_saved, net, net_input, x_init = models.unnp_fit(
        net,
        num_channels,
        A_var,
        y_var,
        num_iter_outer=600,
        num_init_iter_inner=10,
        optim_outer="sgd",
        optim_inner="adam",
        lr_outer=5,
        lr_inner=0.0001,
        find_best=False,
        gpu_flag=GPU_FLAG,
    )

    # Print out the total running time
    t1 = time.time()
    print()
    print(f"Total running time: {t1-t0:.2f} s, GPU used: {GPU_FLAG}")

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)

    ax1.plot(loss_t)
    ax1.set_xlabel("Optimizer iteration")
    ax1.set_ylabel("Loss")

    # Check how the reconstructed image looks like
    out_img_np = net(net_input_saved.type(dtype)).data.cpu().numpy()[0]
    out_img_np = out_img_np[0, :, :]

    print(f"NMSE: {helpers.compute_nmse(x_im, out_img_np)}")

    # Plot the images
    fig2, ax2 = visualize.plot_images((x_im, out_img_np))

    plt.show()


if COMPARE_LOSS_DENOM_SQUARE:
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_test.csv", normalize=True, max_rows=10
    )

    x_im = images[0]

    A = cs_func.create_A(500, 784, seed=1)
    y = cs_func.calc_y(A, x_im.flatten())

    # Move the data over to the set device
    A_var = torch.from_numpy(A).float().to(device)
    y_var = torch.from_numpy(y).float().to(device)

    # Set the parameters for the decoder network
    num_channels = [25, 15, 10]
    output_depth = 1  # number of output channels

    # Create the network
    net = models.DecoderNet(
        num_output_channels=output_depth,
        num_channels_up=num_channels,
        leakyrelu_flag=False,
    ).type(dtype)

    net2 = copy.deepcopy(net)

    # Move the net to the correct device
    net.to(device)

    # Store the time before the fitting
    t0 = time.time()

    # Run the fitting
    loss_t, net_input_saved, net, net_input, x_init = models.unnp_fit(
        net,
        num_channels,
        A_var,
        y_var,
        num_iter_outer=600,
        num_init_iter_inner=10,
        optim_outer="sgd",
        optim_inner="adam",
        lr_outer=5,
        lr_inner=0.0001,
        square_denom_loss=False,
        find_best=False,
        gpu_flag=GPU_FLAG,
    )

    # Run the fitting with square_denom_loss = True
    (
        loss_t_square,
        net_input_saved_square,
        net_square,
        net_input_square,
        x_init_square,
    ) = models.unnp_fit(
        net2,
        num_channels,
        A_var,
        y_var,
        num_iter_outer=600,
        num_init_iter_inner=10,
        optim_outer="sgd",
        optim_inner="adam",
        lr_outer=5,
        lr_inner=0.0001,
        square_denom_loss=True,
        find_best=False,
        gpu_flag=GPU_FLAG,
    )

    # Print out the total running time
    t1 = time.time()
    print()
    print(f"Total running time: {t1-t0:.2f} s, GPU used: {GPU_FLAG}")

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)

    ax1.plot(loss_t)
    ax1.set_xlabel("Optimizer iteration")
    ax1.set_ylabel("Loss")

    # Check how the reconstructed images looks like
    out_img_np = net(net_input_saved.type(dtype)).data.cpu().numpy()[0]
    out_img_np = out_img_np[0, :, :]

    out_img_np_square = net2(net_input_saved_square.type(dtype)).data.cpu().numpy()[0]
    out_img_np_square = out_img_np_square[0, :, :]

    print(f"NMSE: {helpers.compute_nmse(x_im, out_img_np)}")

    # Plot the images
    fig2, ax2 = visualize.plot_images(
        (x_im, out_img_np, out_img_np_square),
        figsize=(8, 3),
        add_cbar=True,
    )

    path_savefig = "data\\figures\\"

    # Create the directory if it does not exist yet
    if not Path(path_savefig).exists():
        Path(path_savefig).mkdir(parents=True)

    if path_savefig is not None:
        fig2.savefig(path_savefig + "compare_square_loss.pdf", dpi=200)
        plt.close(fig2)

if UNNP_TEST_NUM_M:
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_test.csv", normalize=True, max_rows=None
    )

    # The settings for the for loops
    seeds = helpers.get_seeds()
    idx_row_images = helpers.get_idx_row_images()
    num_m = np.array([25, 100, 200, 500, 1000, 1500])
    NUM_RESTARTS = 3

    # Set the parameters for the decoder network
    num_channels = [25, 15, 10]
    output_depth = 1  # number of output channels

    mse = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0], NUM_RESTARTS),
        dtype=np.float64,
    )
    nmse = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0], NUM_RESTARTS),
        dtype=np.float64,
    )
    psnr = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0], NUM_RESTARTS),
        dtype=np.float64,
    )

    for i in tqdm(
        range(seeds.shape[0]), "Progress seeds loop"
    ):  # Loop over all the seeds
        seed = seeds[i]

        # Loop over all the different number of measurments
        for j in tqdm(range(num_m.shape[0]), "Progress num m loop"):
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

                # Move the data over to the set device
                A_var = torch.from_numpy(A).float().to(device)
                y_var = torch.from_numpy(y).float().to(device)

                # Do for every image a numer of random restarts
                for p in range((NUM_RESTARTS)):
                    # Create the network
                    net = models.DecoderNet(
                        num_output_channels=output_depth,
                        num_channels_up=num_channels,
                        leakyrelu_flag=False,
                    ).type(dtype)

                    # Move the net to the correct device
                    net.to(device)

                    # Reconstruct with UNNP
                    loss_t, net_input_saved, net, net_input, x_init = models.unnp_fit(
                        net,
                        num_channels,
                        A_var,
                        y_var,
                        num_iter_outer=600,
                        num_init_iter_inner=10,
                        optim_outer="sgd",
                        optim_inner="adam",
                        lr_outer=5,
                        lr_inner=0.0001,
                        find_best=False,
                        square_denom_loss=False,
                        verbose=False,
                        gpu_flag=GPU_FLAG,
                    )

                    x_hat = net(net_input_saved.type(dtype)).data.cpu().numpy()[0]
                    x_hat = x_hat[0, :, :]
                    x_hat = np.reshape(x_hat, (28, 28))

                    # Compute the metrics
                    mse[i, j, k, p] = helpers.compute_mse(x_im, x_hat)
                    nmse[i, j, k, p] = helpers.compute_nmse(x_im, x_hat)

                    x_hat_norm = helpers.normalize(x_hat)
                    psnr[i, j, k, p] = helpers.compute_psnr(x_im, x_hat_norm)

    path_to_data = f"data\\unnp\\metrics_num_m\\raw\\"

    # If the path does not yet exists, create it
    if not Path(path_to_data).exists():
        Path(path_to_data).mkdir(parents=True)

    # Save all the different data arrays
    process_data.save_arr(path_to_data + "num_m.npy", num_m)
    process_data.save_arr(path_to_data + "mse.npy", mse)
    process_data.save_arr(path_to_data + "nmse.npy", nmse)
    process_data.save_arr(path_to_data + "psnr.npy", psnr)


if PROCESS_DATA_UNNP_TEST_NUM_M:
    path_to_data_raw = f"data\\unnp\\metrics_num_m\\raw\\"
    if not Path(path_to_data_raw).exists():
        raise FileNotFoundError("The data does not exist, first generate the data.")

    path_to_data_processed = f"data\\unnp\\metrics_num_m\\processed\\"
    if not Path(path_to_data_processed).exists():
        Path(path_to_data_processed).mkdir(parents=True)

    # Load all the different data arrays
    num_m = process_data.load_arr(path_to_data_raw + "num_m.npy")
    mse = process_data.load_arr(path_to_data_raw + "mse.npy")
    nmse = process_data.load_arr(path_to_data_raw + "nmse.npy")
    psnr = process_data.load_arr(path_to_data_raw + "psnr.npy")

    # Take the mean over all the random restarts
    mse_mean_restarts = np.mean(mse, axis=3)
    nmse_mean_restarts = np.mean(nmse, axis=3)
    psnr_mean_restarts = np.mean(psnr, axis=3)

    # Take the mean over all the seeds for every m and every image
    mse_mean_seeds = np.mean(mse_mean_restarts, axis=0)
    nmse_mean_seeds = np.mean(nmse_mean_restarts, axis=0)
    psnr_mean_seeds = np.mean(psnr_mean_restarts, axis=0)

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


if PLOT_RESULTS_UNNP_TEST_NUM_M:
    path_to_data_processed = f"data\\unnp\\metrics_num_m\\processed\\"

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


if UNNP_TEST_NUM_M_LEAKYRELU:
    labels, images = process_data.load_mnist_data(
        "data\\raw\\mnist_test.csv", normalize=True, max_rows=None
    )

    # The settings for the for loops
    seeds = helpers.get_seeds()
    idx_row_images = helpers.get_idx_row_images()
    num_m = np.array([25, 100, 200, 500, 1000, 1500])
    NUM_RESTARTS = 3

    # Set the parameters for the decoder network
    num_channels = [25, 15, 10]
    output_depth = 1  # number of output channels

    mse = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0], NUM_RESTARTS),
        dtype=np.float64,
    )
    nmse = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0], NUM_RESTARTS),
        dtype=np.float64,
    )
    psnr = np.zeros(
        (seeds.shape[0], num_m.shape[0], idx_row_images.shape[0], NUM_RESTARTS),
        dtype=np.float64,
    )

    for i in tqdm(
        range(seeds.shape[0]), "Progress seeds loop"
    ):  # Loop over all the seeds
        seed = seeds[i]

        # Loop over all the different number of measurments
        for j in tqdm(range(num_m.shape[0]), "Progress num m loop"):
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

                # Move the data over to the set device
                A_var = torch.from_numpy(A).float().to(device)
                y_var = torch.from_numpy(y).float().to(device)

                # Do for every image a numer of random restarts
                for p in range((NUM_RESTARTS)):
                    # Create the network
                    net = models.DecoderNet(
                        num_output_channels=output_depth,
                        num_channels_up=num_channels,
                        leakyrelu_flag=True,
                    ).type(dtype)

                    # Move the net to the correct device
                    net.to(device)

                    # Reconstruct with UNNP
                    loss_t, net_input_saved, net, net_input, x_init = models.unnp_fit(
                        net,
                        num_channels,
                        A_var,
                        y_var,
                        num_iter_outer=600,
                        num_init_iter_inner=10,
                        optim_outer="sgd",
                        optim_inner="adam",
                        lr_outer=5,
                        lr_inner=0.0001,
                        find_best=False,
                        square_denom_loss=False,
                        verbose=False,
                        gpu_flag=GPU_FLAG,
                    )

                    x_hat = net(net_input_saved.type(dtype)).data.cpu().numpy()[0]
                    x_hat = x_hat[0, :, :]
                    x_hat = np.reshape(x_hat, (28, 28))

                    # Compute the metrics
                    mse[i, j, k, p] = helpers.compute_mse(x_im, x_hat)
                    nmse[i, j, k, p] = helpers.compute_nmse(x_im, x_hat)

                    x_hat_norm = helpers.normalize(x_hat)
                    psnr[i, j, k, p] = helpers.compute_psnr(x_im, x_hat_norm)

    path_to_data = f"data\\unnp\\metrics_num_m_leakyrelu\\raw\\"

    # If the path does not yet exists, create it
    if not Path(path_to_data).exists():
        Path(path_to_data).mkdir(parents=True)

    # Save all the different data arrays
    process_data.save_arr(path_to_data + "num_m.npy", num_m)
    process_data.save_arr(path_to_data + "mse.npy", mse)
    process_data.save_arr(path_to_data + "nmse.npy", nmse)
    process_data.save_arr(path_to_data + "psnr.npy", psnr)


if PROCESS_DATA_UNNP_TEST_NUM_M_LEAKYRELU:
    path_to_data_raw = f"data\\unnp\\metrics_num_m_leakyrelu\\raw\\"
    if not Path(path_to_data_raw).exists():
        raise FileNotFoundError("The data does not exist, first generate the data.")

    path_to_data_processed = f"data\\unnp\\metrics_num_m_leakyrelu\\processed\\"
    if not Path(path_to_data_processed).exists():
        Path(path_to_data_processed).mkdir(parents=True)

    # Load all the different data arrays
    num_m = process_data.load_arr(path_to_data_raw + "num_m.npy")
    mse = process_data.load_arr(path_to_data_raw + "mse.npy")
    nmse = process_data.load_arr(path_to_data_raw + "nmse.npy")
    psnr = process_data.load_arr(path_to_data_raw + "psnr.npy")

    # Take the mean over all the random restarts
    mse_mean_restarts = np.mean(mse, axis=3)
    nmse_mean_restarts = np.mean(nmse, axis=3)
    psnr_mean_restarts = np.mean(psnr, axis=3)

    # Take the mean over all the seeds for every m and every image
    mse_mean_seeds = np.mean(mse_mean_restarts, axis=0)
    nmse_mean_seeds = np.mean(nmse_mean_restarts, axis=0)
    psnr_mean_seeds = np.mean(psnr_mean_restarts, axis=0)

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


if PLOT_RESULTS_UNNP_TEST_NUM_M_LEAKYRELU:
    path_to_data_processed = f"data\\unnp\\metrics_num_m_leakyrelu\\processed\\"

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


if PLOT_RESULTS_COMPARE_UNNP_TEST_NUM_M_RELU_VS_LEAKYRELU:
    path_to_data_processed_relu = f"data\\unnp\\metrics_num_m\\processed\\"
    path_to_data_processed_leakyrelu = (
        f"data\\unnp\\metrics_num_m_leakyrelu\\processed\\"
    )

    # Check if the paths exist
    if not Path(path_to_data_processed_relu).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )
    if not Path(path_to_data_processed_leakyrelu).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    # Load all the different data arrays for ReLU results
    num_m_relu = process_data.load_arr(path_to_data_processed_relu + "num_m.npy")
    mse_mean_relu = process_data.load_arr(path_to_data_processed_relu + "mse_mean.npy")
    mse_std_relu = process_data.load_arr(path_to_data_processed_relu + "mse_std.npy")
    nmse_mean_relu = process_data.load_arr(
        path_to_data_processed_relu + "nmse_mean.npy"
    )
    nmse_std_relu = process_data.load_arr(path_to_data_processed_relu + "nmse_std.npy")
    psnr_mean_relu = process_data.load_arr(
        path_to_data_processed_relu + "psnr_mean.npy"
    )
    psnr_std_relu = process_data.load_arr(path_to_data_processed_relu + "psnr_std.npy")

    # Load all the different data arrays for LeakyReLU results
    num_m_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "num_m.npy"
    )
    mse_mean_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "mse_mean.npy"
    )
    mse_std_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "mse_std.npy"
    )
    nmse_mean_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "nmse_mean.npy"
    )
    nmse_std_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "nmse_std.npy"
    )
    psnr_mean_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "psnr_mean.npy"
    )
    psnr_std_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "psnr_std.npy"
    )

    # Create the figure
    fig4, axs4 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axs4[0].errorbar(
        num_m_relu,
        mse_mean_relu,
        yerr=mse_std_relu,
        fmt="--o",
        capsize=3,
        label="ReLU",
    )
    axs4[0].errorbar(
        num_m_leakyrelu,
        mse_mean_leakyrelu,
        yerr=mse_std_relu,
        fmt="--o",
        capsize=3,
        label="LeakyReLU",
    )
    axs4[0].set_xlabel("Number of measurements (m)")
    axs4[0].set_ylabel("MSE")
    axs4[0].grid(True)

    axs4[1].errorbar(
        num_m_relu,
        nmse_mean_relu,
        yerr=nmse_std_relu,
        fmt="--o",
        capsize=3,
        label="ReLU",
    )
    axs4[1].errorbar(
        num_m_leakyrelu,
        nmse_mean_leakyrelu,
        yerr=nmse_std_leakyrelu,
        fmt="--o",
        capsize=3,
        label="LeakyReLU",
    )
    axs4[1].set_xlabel("Number of measurements (m)")
    axs4[1].set_ylabel("NMSE")
    axs4[1].grid(True)

    axs4[2].errorbar(
        num_m_relu,
        psnr_mean_relu,
        yerr=psnr_std_relu,
        fmt="--o",
        capsize=3,
        label="ReLU",
    )
    axs4[2].errorbar(
        num_m_leakyrelu,
        psnr_mean_leakyrelu,
        yerr=psnr_std_leakyrelu,
        fmt="--o",
        capsize=3,
        label="LeakyReLU",
    )
    axs4[2].set_xlabel("Number of measurements (m)")
    axs4[2].set_ylabel("PSNR")
    axs4[2].grid(True)

    fig4.tight_layout()

    plt.show()

if PLOT_RESULTS_UNNP_TEST_NUM_M_LEAKYRELU_OLNY_NMSE:
    path_to_data_processed_relu = f"data\\unnp\\metrics_num_m\\processed\\"
    path_to_data_processed_leakyrelu = (
        f"data\\unnp\\metrics_num_m_leakyrelu\\processed\\"
    )

    # Check if the paths exist
    if not Path(path_to_data_processed_relu).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )
    if not Path(path_to_data_processed_leakyrelu).exists():
        raise FileNotFoundError(
            "The data does not exist, first generate/process the data."
        )

    # Load all the different data arrays for ReLU results
    num_m_relu = process_data.load_arr(path_to_data_processed_relu + "num_m.npy")
    nmse_mean_relu = process_data.load_arr(
        path_to_data_processed_relu + "nmse_mean.npy"
    )
    nmse_std_relu = process_data.load_arr(path_to_data_processed_relu + "nmse_std.npy")
    # Load all the different data arrays for LeakyReLU results
    num_m_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "num_m.npy"
    )
    nmse_mean_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "nmse_mean.npy"
    )
    nmse_std_leakyrelu = process_data.load_arr(
        path_to_data_processed_leakyrelu + "nmse_std.npy"
    )

    # Create the figure
    fig4, axs4 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))

    axs4.errorbar(
        num_m_relu,
        nmse_mean_relu,
        yerr=nmse_std_relu,
        fmt="--o",
        capsize=3,
        label="ReLU",
    )
    axs4.errorbar(
        num_m_leakyrelu,
        nmse_mean_leakyrelu,
        yerr=nmse_std_leakyrelu,
        fmt="--o",
        capsize=3,
        label="LeakyReLU",
    )
    axs4.set_xlabel("Number of measurements (m)")
    axs4.set_ylabel("NMSE")
    axs4.grid(True)

    fig4.tight_layout()
    plt.legend(loc="best", title="UNNP type")

    # plt.show()

    path_savefig = "data\\figures\\"

    # Create the directory if it does not exist yet
    if not Path(path_savefig).exists():
        Path(path_savefig).mkdir(parents=True)

    if path_savefig is not None:
        fig4.savefig(path_savefig + "relu_vs_leakyrelu.pdf", dpi=200)
        plt.close(fig4)
