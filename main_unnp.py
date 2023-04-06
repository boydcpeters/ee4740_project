import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import models
import cs_func
import process_data
import visualize
import helpers

import torch
import torch.nn as nn

# First determine whether the model should be run on the CPU or GPU,
# CPU seems to be quicker (since the model is relatively small)
GPU_FLAG = False
dtype, device = models.get_dtype_device(GPU_FLAG)

TEST_RUN_NET_FLAG = False
COMPARE_LOSS_DENOM_SQUARE_FLAG = False
COMPARE_LOSS_RELU_VS_LEAKYRELU = True

if TEST_RUN_NET_FLAG:
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
        find_best=True,
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


if COMPARE_LOSS_DENOM_SQUARE_FLAG:
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
        ("Original image", "Denominator loss", "Denominator loss squared"),
        figsize=(9, 3),
        add_cbar=True,
    )

    plt.show()

if COMPARE_LOSS_RELU_VS_LEAKYRELU:
    pass
    # TODO: write code here to compare RELU VS LEAKYRELU


# labels, images = process_data.load_mnist_data(
#     "data\\raw\\mnist_train.csv", normalize=True, max_rows=10
# )

# x_im = images[1]

# A = cs_func.create_A(200, 784, seed=1)
# y = cs_func.calc_y(A, x_im.flatten())
# # y = A @ x_im.flatten()

# # print(y)

# num_channels = [25, 15, 10]
# output_depth = 1  # number of output channels

# A_var = torch.from_numpy(A).float().to(device)
# y_var = torch.from_numpy(y).float().to(device)

# temp1 = y @ (A @ x_im.flatten())
# temp2 = np.linalg.norm(x_im)  # ** 2

# perfect_loss = -temp1 / temp2

# print("so the perfect loss is not very far away")
# print(f"perfect loss: {perfect_loss}")


# net = models.DecoderNet(
#     num_output_channels=output_depth, num_channels_up=num_channels, leakyrelu_flag=False
# ).type(dtype)

# # Move the net to the correct device
# net.to(device)

# t0 = time.time()

# loss_t, net_input_saved, net, net_input, x_init = models.unnp_fit(
#     net,
#     num_channels,
#     A_var,
#     y_var,
#     num_iter_outer=600,
#     num_init_iter_inner=10,
#     optim_outer="sgd",
#     optim_inner="adam",
#     lr_outer=5,
#     lr_inner=0.0001,
#     lr_decay_epoch=500,
#     find_best=False,
#     gpu_flag=GPU_FLAG,
# )
# t1 = time.time()

# print()
# print(f"Total running time: {t1-t0:.2f} s, GPU: {GPU_FLAG}")


# fig1, ax1 = plt.subplots(nrows=1, ncols=1)

# ax1.plot(loss_t)
# ax1.set_xlabel("optimizer iteration")
# ax1.set_ylabel("loss")

# # Check how the reconstructed image looks like
# out_img_np = net(net_input_saved.type(dtype)).data.cpu().numpy()[0]
# out_img_np = out_img_np[0, :, :]

# print(f"nmse: {helpers.compute_nmse(x_im, out_img_np)}")

# # Plot the images
# fig2, ax2 = visualize.plot_images((x_im, out_img_np))

# plt.show()
