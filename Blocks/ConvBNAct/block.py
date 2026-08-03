import torch
import torch.nn as nn

from ..BatchNorm.bnmodule import BatchNorm2d
from ..Conv.block import Conv2D
from ..DepthwiseConv.block import DepthwiseConv
from ..PointwiseConv.block import PointwiseConv
from ..ReLU.block import ReLUBlock


ACTIVATIONS = {
    "relu": ReLUBlock.apply,
}


class ConvBNAct(nn.Module):


    def __init__(self, c_in, c_out, conv_type="standard", kernel_size=None,
                 stride=1, padding=None, act="relu", use_bn=True):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.conv_type = conv_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.act = act

        if conv_type == "standard":
            weight = torch.empty(c_out, c_in, kernel_size, kernel_size)
        elif conv_type == "depthwise":
            weight = torch.empty(c_in, 1, kernel_size, kernel_size)
        else:
            weight = torch.empty(c_out, c_in)

        self.W = nn.Parameter(weight)
        nn.init.kaiming_normal_(self.W, mode="fan_out", nonlinearity="relu")

        if use_bn:
            self.bn = BatchNorm2d(c_out)
            self.register_parameter("b", None)  # batchnorm cancels it anyway
        else:
            self.bn = None
            self.b = nn.Parameter(torch.zeros(c_out))

    def forward(self, X):
        if X.shape[1] != self.c_in:
            raise RuntimeError(
                f"Error: expected {self.c_in} input channels, got {X.shape[1]}"
            )

        if self.conv_type == "standard":
            out = Conv2D.apply(X, self.W, self.b, self.stride, self.padding)
        elif self.conv_type == "depthwise":
            out = DepthwiseConv.apply(X, self.W, self.b, self.stride, self.padding)
        else:
            out = PointwiseConv.apply(X, self.W, self.b)

        if self.bn is not None:
            out = self.bn(out)
        if self.act is not None:
            out = ACTIVATIONS[self.act](out)

        return out

    def extra_repr(self):
        pad = "same" if self.padding is None else self.padding
        return (f"{self.c_in} -> {self.c_out}, {self.conv_type}, k={self.kernel_size}, "
                f"s={self.stride}, pad={pad}, bn={self.bn is not None}, act={self.act}")
