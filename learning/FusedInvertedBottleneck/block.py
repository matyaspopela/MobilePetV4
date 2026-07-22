import torch
import torch.nn as nn
from ..ConvBlock.block import Conv2D
from ..PointwiseConv.block import PointwiseConv

class FusedInvertedBottleneck(nn.Module):
    def __init__(self, c_in, c_mid, c_out, kernel_size=(3,3),stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.W_fused = nn.Parameter(torch.randn(c_mid, c_in, *kernel_size) * 0.01)
        self.W_project = nn.Parameter(torch.randn(c_out, c_mid) * 0.01)

    def forward(self, X):
        fused = Conv2D.apply(X, self.W_fused, stride=self.stride, padding=self.padding)
        projected = PointwiseConv.apply(fused, self.W_project)
        return projected