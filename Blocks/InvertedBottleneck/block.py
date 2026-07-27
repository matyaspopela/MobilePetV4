import torch
import torch.nn as nn

from ..PointwiseConv.block import PointwiseConv
from ..DepthwiseConv.block import DepthwiseConv



class InvertedBottleneck(nn.Module):
    def __init__(self, c_in, c_mid, c_out, depth_kernel_size = (3, 3), depth_stride=1, depth_padding=0):
        super().__init__()
        #learnable params
        self.W_expand = nn.Parameter(torch.randn(c_mid, c_in) * 0.01)
        self.b_expand = nn.Parameter(torch.randn(c_mid) * 0.01)

        self.W_depth = nn.Parameter(torch.randn(c_mid, 1, *depth_kernel_size) * 0.01)
        self.b_depth = nn.Parameter(torch.randn(c_mid) * 0.01)

        self.W_project = nn.Parameter(torch.randn(c_out, c_mid) * 0.01)
        self.b_project = nn.Parameter(torch.randn(c_out) * 0.01)
        #hyperparams
        self.depth_stride = depth_stride
        self.depth_padding = depth_padding

    def forward(self, X):
        expanded = PointwiseConv.apply(X, self.W_expand, self.b_expand)
        depthwise = DepthwiseConv.apply(expanded, self.W_depth, self.b_depth, self.depth_stride, self.depth_padding)
        projected = PointwiseConv.apply(depthwise, self.W_project, self.b_project)
        return projected