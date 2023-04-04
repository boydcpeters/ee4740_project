# TODO

For every algorithm

Do this multiple times (with the same images) {

- multiple A matrices (25, 100, 200, 500 minimum)
- do the mse, nmse, and psnr calculation for multiple images
}

seeds = helpers.get_seeds()
idx_row_images = helpers.gsset_idx_row_images()

for seed in seeds:
    for m in [25, 100, 200, 500]:

        A = create_A(m, 784, seed=seed)

        for im in images:

            y = A @ im

            x_hat = biht(A, im)

            mse = ...
            nmse = ...
            psnr = ...

mse_mean = ...
nmse_mean = ...
psnr_mean = ...

mse_std = ...
nmse_std = ...
psnr_std = ...

plot(m, mse_mean + add confidence interval based on std)
...
