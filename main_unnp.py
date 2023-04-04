import os
import numpy as np
import matplotlib.pyplot as plt
import models
import cs_func
import process_data
import visualize

import torch
import torch.nn as nn

GPU_FLAG = False
dtype, device = models.get_dtype_device(GPU_FLAG)


labels, images = process_data.load_mnist_data(
    "data\\raw\\mnist_train.csv", normalize=True, max_rows=10
)

x_im = images[0]

A = cs_func.create_A(500, 784, seed=1)
y = cs_func.calc_y(A, x_im.flatten())
# y = A @ x_im.flatten()

num_channels = [25, 15, 10]
output_depth = 1  # number of output channels

A_var = torch.from_numpy(A).float().to(device)
y_var = torch.from_numpy(y).float().to(device)

net = models.DecoderNet(
    num_output_channels=output_depth, num_channels_up=num_channels, leakyrelu_flag=False
).type(dtype)

# Move the net to the correct device
net.to(device)

loss_t, net_input_saved, net, net_input, x_init = models.unnp_fit(
    net,
    num_channels,
    A_var,
    y_var,
    num_iter_outer=1000,
    num_iter_inner=30,
    lr_outer=0.35,
    lr_inner=0.03,
    gpu_flag=GPU_FLAG,
)

fig1, ax1 = plt.subplots(nrows=1, ncols=1)

ax1.semilogx(loss_t)
ax1.set_xlabel("optimizer iteration")
ax1.set_ylabel("loss")

# Check how the reconstructed image looks like
out_img_np = net(net_input_saved.type(dtype)).data.cpu().numpy()[0]
out_img_np = out_img_np[0, :, :]

# Plot the images
fig2, ax2 = visualize.plot_images((x_im, out_img_np))

plt.show()
