import data_processing
import visualize
import cs_func

import numpy as np
import matplotlib.pyplot as plt


labels, images = data_processing.load_mnist_data("data\mnist_test.csv")

print(labels)

a = images[0]

# fig1, ax1 = visualize.plot_image(images[0])

fig1, axs = visualize.plot_images(images[:8])

x = images[0]
x = x.flatten()

A = cs_func.create_A(100, 784)
y = cs_func.calc_y(A, x)
print(np.sum(A[:, 0] ** 2))
print(y)
# plt.show()
