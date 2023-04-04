import process_data
import visualize
import cs_func
import models
from helpers import *

import numpy as np
import matplotlib.pyplot as plt


labels, images = process_data.load_mnist_data(
    "data\\raw\\mnist_test.csv", normalize=True, max_rows=10
)

a = images[0]

# fig1, ax1 = visualize.plot_image(images[0])

# fig1, axs = visualize.plot_images(images[:8])

x_im = images[4]
x = x_im.flatten()

A = cs_func.create_A(500, 784, seed=2)
y = cs_func.calc_y(A, x)
print(np.sum(A[:, 0] ** 2))
# print(y)

# x_hat = models.convex(A, y)

# x_hat = models.rfpi(A, y, alpha=0.05, max_iter=100, verbose=True)
# # TODO: fix l2-mode, it is currently not working
x_hat = models.biht(A, y, 200, max_iter=3000, mode="l1", verbose=True)

# # print("ADAP")
# # x_hat_adap = models.biht_adap(A, y, 300, max_iter=3000, mode="l1", verbose=True)

# x_hat = np.reshape(x_hat, (28, 28))
# # x_hat_adap = np.reshape(x_hat_adap, (28, 28))


# # vis = np.concatenate(, axis=2)

# # print(vis.shape)
x_hat = np.reshape(x_hat, (28, 28))

print(f"mse normal: {compute_mse(x_im, x_hat)}")
print(f"nmse normal: {compute_nmse(x_im, x_hat)}")


# print(f"mse adap: {compute_mse(x_im, x_hat_adap)}")
# print(f"nmse adap: {compute_nmse(x_im, x_hat_adap)}")

fig2, ax2 = visualize.plot_images((x_im, x_hat))  # , x_hat_adap))
plt.show()
# plt.show()
