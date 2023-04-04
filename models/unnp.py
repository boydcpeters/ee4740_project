from typing import Union, List, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import torch.optim
from tqdm import tqdm


def get_dtype_device(gpu_flag):
    if gpu_flag:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"num GPUs: {torch.cuda.device_count()}")
        dtype = torch.cuda.FloatTensor
        device = "cuda"
        if torch.cuda.device_count() == 0:
            dtype = torch.FloatTensor
            device = "cpu"
    else:
        dtype = torch.FloatTensor
        device = "cpu"

    return dtype, device


if torch.cuda.device_count() == 0:
    dtype = torch.FloatTensor
    device = "cpu"
else:
    dtype = torch.cuda.FloatTensor
    device = "cuda"


def custom_loss(A: torch.tensor, x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """_summary_

    Parameters
    ----------
    A : torch.tensor
        _description_
    x : torch.tensor
        _description_
    y : torch.tensor
        _description_

    Returns
    -------
    torch.tensor
        _description_
    """
    a = y.shape
    aa = A.shape
    aaa = x.shape

    return -(torch.matmul(y, torch.matmul(A, x.reshape(x.numel(), 1)))) / (
        torch.norm(x) ** 2
    )


class DecoderNet(nn.Module):
    def __init__(
        self,
        num_output_channels,
        num_channels_up,
        upsample_mode: str = "bilinear",
        leakyrelu_flag: bool = False,
    ):
        super(DecoderNet, self).__init__()

        self.decoder = nn.Sequential()

        n_chn = len(num_channels_up)  # number of channels

        for i in range(n_chn - 1):
            module_name = f"dconv{i}"
            self.decoder.add_module(
                module_name,
                nn.Conv2d(
                    num_channels_up[i],
                    num_channels_up[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

            # Add the ReLU/LeakReLU layer
            if leakyrelu_flag:
                module_name = f"dleakyrelu{i}"
                self.decoder.add_module(module_name, nn.LeakyReLU())
            else:
                module_name = f"drelu{i}"
                self.decoder.add_module(module_name, nn.ReLU())

            # Add the upsampling layer
            module_name = f"dups{i}"
            self.decoder.add_module(
                module_name, nn.Upsample(scale_factor=2, mode=upsample_mode)
            )

            # Add the batch normalization layer
            module_name = f"dbn{i}"
            self.decoder.add_module(
                module_name, nn.BatchNorm2d(num_channels_up[i + 1], affine=True)
            )

        # Add the final convolutional layer
        module_name = f"dconv{i+1}"
        self.decoder.add_module(
            module_name,
            nn.Conv2d(
                num_channels_up[-1],
                num_output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        # Finally, add the sigmoid layer
        self.decoder.add_module("dsig", nn.Sigmoid())

    def forward(self, x):
        return self.decoder(x)


def unnp_fit(
    net: DecoderNet,
    num_channels: Union[List[int], Tuple[int]],
    A: np.ndarray,
    y: np.ndarray,
    net_input: np.ndarray = None,
    net_input_distr: str = "uniform",
    optim_outer: str = "sgd",
    optim_inner: str = "sgd",
    lr_outer: float = 0.01,
    lr_inner: float = 0.02,
    num_iter_outer: int = 5000,
    num_iter_inner: int = 100,
    out_channels: int = 1,
    weight_decay: float = 0.0,
    verbose: bool = True,
    gpu_flag: bool = False,
):
    print("Start the fit") if verbose else None

    dtype, device = get_dtype_device(gpu_flag)

    m, n = A.shape

    if net_input is not None:
        print("Input for the network is provided")
    else:
        # Feed uniform noise into the network

        # Calculate how many times the latent code will be upsampled
        total_upsample = 2 ** (len(num_channels) - 1)

        w = np.sqrt(int(n / out_channels))
        width = int(w / total_upsample)
        height = int(w / total_upsample)

        shape = [1, num_channels[0], width, height]
        print(f"The shape of latent code: {shape}") if verbose else None

        # Create the variable for storing of the latent code
        net_input = Variable(torch.zeros(shape))

        # Perform the random sampling to fill the data
        if net_input_distr == "uniform":
            net_input.data.uniform_()
        elif net_input_distr == "gaussian":
            net_input.data.normal_()

        net_input.data *= 1.0 / 10.0

    # We don't optimze over the input so no gradient required
    # net_input.requires_grad = False

    # Store the initial input values
    net_input_store = net_input.data.clone()

    p = [x for x in net.decoder.parameters()]  # List of all the weights

    loss_wrt_truth = np.zeros(num_iter_outer)

    if optim_inner == "sgd":
        print("Optimize decoder with SGD")
        optimizer_inner = torch.optim.SGD(
            p, lr=lr_inner, momentum=0.9, weight_decay=weight_decay
        )
    elif optim_inner == "adam":
        print("Optimize decoder with Adam")
        optimizer_inner = torch.optim.Adam(p, lr=lr_inner, weight_decay=weight_decay)

    print("Optimizing with projected gradient descent...")

    x = Variable(torch.zeros([out_channels, int(w), int(w)]))
    x = x.to(device)

    test = net_input.type(dtype)
    x.data = net(net_input.type(dtype))

    # The initial x tensor
    x_init = x.data.clone()

    x.requires_grad = True
    x.retain_grad()

    xvar = [x]

    if optim_outer == "sgd":
        optimizer_outer = torch.optim.SGD(
            xvar, lr=lr_outer, momentum=0.9, weight_decay=weight_decay
        )
    elif optim_outer == "adam":
        optimizer_outer = torch.optim.Adam(xvar, lr=lr_outer, weight_decay=weight_decay)

    # Create MSE loss criterion
    mse = torch.nn.MSELoss()

    for i in tqdm(range(num_iter_outer)):
        with torch.no_grad():
            loss_pre = mse(torch.matmul(A, x.reshape(x.numel(), 1)), y)

        # Gradient step for the least squares problem
        optimizer_outer.zero_grad()

        output = torch.matmul(A, x.reshape(x.numel(), 1))
        loss_outer = custom_loss(
            A,
            x,
            y,
        )
        loss_outer.backward()
        optimizer_outer.step()

        # Store the loss for an iteration
        loss_wrt_truth[i] = loss_outer.item()

        print("Iteration %05d   Train loss %f " % (i, loss_wrt_truth[i]), "\r", end="")

        for j in range(num_iter_inner):
            optimizer_inner.zero_grad()
            out = net(net_input.type(dtype))
            loss_inner = mse(out, x)
            loss_inner.backward()
            optimizer_inner.step()

        # Project on the learned network
        x.data = net(net_input.type(dtype))

        # TODO: remove loss_updated or use it for best net?
        loss_updated = custom_loss(A, Variable(x.data, requires_grad=True).flatten(), y)

    return loss_wrt_truth, net_input_store, net, net_input, x_init
