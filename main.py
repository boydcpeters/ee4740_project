import process_data
import visualize
import cs_func
import models
from helpers import *

import numpy as np
import matplotlib.pyplot as plt


labels, images = process_data.load_mnist_data(
    "data\mnist_test.csv", normalize=True, max_rows=None
)

print(images.shape)

images_nonzeros = images >= (0.95 / 255.0)

# For every image note the number of nonzeros
s_images = np.zeros(images.shape[0])

for i in range(s_images.shape[0]):
    s_images[i] = np.count_nonzero(images_nonzeros[i, :, :])

print(np.amax(s_images))

unique, count = np.unique(s_images, return_counts=True)

print(f"unique: {unique}, count: {count}")

print(f"mean: {np.sum(unique * count) * (1/np.sum(count))}")

plt.hist(s_images, bins=31, density=True)

plt.show()

# # print(labels)

# a = images[0]

# # fig1, ax1 = visualize.plot_image(images[0])

# # fig1, axs = visualize.plot_images(images[:8])

# x_im = images[4]
# x = x_im.flatten()

# A = cs_func.create_A(1500, 784, seed=2)
# y = cs_func.calc_y(A, x)
# print(np.sum(A[:, 0] ** 2))
# # print(y)

# # TODO: fix l2-mode, it is currently not working
# x_hat = models.biht(A, y, 150, max_iter=3000, mode="l1", verbose=True)

# # print("ADAP")
# # x_hat_adap = models.biht_adap(A, y, 300, max_iter=3000, mode="l1", verbose=True)

# x_hat = np.reshape(x_hat, (28, 28))
# # x_hat_adap = np.reshape(x_hat_adap, (28, 28))


# # vis = np.concatenate(, axis=2)

# # print(vis.shape)

# print(f"mse normal: {compute_mse(x_im, x_hat)}")
# print(f"nmse normal: {compute_nmse(x_im, x_hat)}")


# # print(f"mse adap: {compute_mse(x_im, x_hat_adap)}")
# # print(f"nmse adap: {compute_nmse(x_im, x_hat_adap)}")

# fig2, ax2 = visualize.plot_images((x_im, x_hat))  # , x_hat_adap))
# plt.show()
# # plt.show()
