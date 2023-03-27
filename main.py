import process_data
import visualize
import cs_func
import models

import numpy as np
import matplotlib.pyplot as plt


labels, images = process_data.load_mnist_data("data\mnist_test.csv", normalize=False)

# print(labels)

a = images[0]

# fig1, ax1 = visualize.plot_image(images[0])

# fig1, axs = visualize.plot_images(images[:8])

x_im = images[0]
x = x_im.flatten()

A = cs_func.create_A(1500, 784)
y = cs_func.calc_y(A, x)
print(np.sum(A[:, 0] ** 2))
# print(y)

x_hat = models.biht(A, y, 400, max_iter=3000, mode="l2", verbose=True)

x_hat = np.reshape(x_hat, (28, 28))


# vis = np.concatenate(, axis=2)

# print(vis.shape)

fig2, ax2 = visualize.plot_images((x_im, x_hat))
plt.show()
# plt.show()
