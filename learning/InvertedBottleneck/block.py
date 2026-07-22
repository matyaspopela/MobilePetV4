import torch
import torch.nn as nn

from ..PointwiseConv.block import PointwiseConv
from ..depthwise_separable_conv.block import DepthwiseConv



class InvertedBottleneck(nn.Module):
    def __init__(self, c_in, c_mid, c_out, depth_kernel_size = (3, 3), depth_stride=1, depth_padding=0):
        super().__init__()
        self.W_expand = nn.Parameter(torch.randn(c_mid, c_in) * 0.01)
        self.W_depth = nn.Parameter(torch.randn(c_mid, 1, *depth_kernel_size) * 0.01)
        self.W_project = nn.Parameter(torch.randn(c_out, c_mid) * 0.01)
        self.depth_stride = depth_stride
        self.depth_padding = depth_padding

    def forward(self, X):
        expanded = PointwiseConv.apply(X, self.W_expand)
        depthwise = DepthwiseConv.apply(expanded, self.W_depth, self.depth_stride, self.depth_padding)
        projected = PointwiseConv.apply(depthwise, self.W_project)
        return projected

ib = InvertedBottleneck(c_in=16, c_mid=64, c_out=16, depth_kernel_size=(3,3), depth_padding=1)
X = torch.randn(2, 16, 8, 8, requires_grad=True)
Y = ib(X)
print(Y.shape)          # sanity check: should be (2, 16, 8, 8) if padding keeps spatial size
Y.sum().backward()
print(X.grad.shape)     # should be (2, 16, 8, 8) — confirms the whole chain backprop'd