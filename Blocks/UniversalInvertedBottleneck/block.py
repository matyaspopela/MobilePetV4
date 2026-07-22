import torch
import torch.nn as nn

from ..DepthwiseConv.block import DepthwiseConv
from ..PointwiseConv.block import PointwiseConv

class UniversalInvertedBottleneck(nn.Module):
    def __init__(self, c_in, c_mid, c_out,
                 dw_before=False,
                 dw_after=True,
                 dw_before_kernel=(3,3),
                 dw_before_padding=0,
                 dw_before_stride=1,
                 dw_after_kernel=(3,3),
                 dw_after_padding=0,
                 dw_after_stride=1):

        super().__init__()

        # learnable
        if dw_before:
            self.W_dw_before = nn.Parameter(torch.randn(c_in, 1, *dw_before_kernel) * 0.01)
            self.b_dw_before = nn.Parameter(torch.randn(c_in) * 0.01)
            self.dw_before_padding = dw_before_padding
            self.dw_before_stride = dw_before_stride
        else:
            self.register_parameter("W_dw_before", None)

        self.W_expand = nn.Parameter(torch.randn(c_mid, c_in) * 0.01)
        self.b_expand = nn.Parameter(torch.randn(c_mid) * 0.01)

        if dw_after:
            self.W_dw_after = nn.Parameter(torch.randn(c_mid, 1, *dw_after_kernel) * 0.01)
            self.b_dw_after = nn.Parameter(torch.randn(c_mid) * 0.01)
            self.dw_after_padding = dw_after_padding
            self.dw_after_stride = dw_after_stride
        else:
            self.register_parameter("W_dw_after", None)

        self.W_project = nn.Parameter(torch.randn(c_out, c_mid) * 0.01)
        self.b_project = nn.Parameter(torch.randn(c_out) * 0.01)

    def forward(self, X):
        if self.W_dw_before is not None:
            depth_before = DepthwiseConv.apply(X, self.W_dw_before, self.b_dw_before, self.dw_before_stride, self.dw_before_padding)
            expand = PointwiseConv.apply(depth_before, self.W_expand, self.b_expand)
        else:
            expand = PointwiseConv.apply(X, self.W_expand, self.b_expand)

        if self.W_dw_after is not None:
            depth_after = DepthwiseConv.apply(expand, self.W_dw_after, self.b_dw_after, self.dw_after_stride, self.dw_after_padding)
            projected = PointwiseConv.apply(depth_after, self.W_project, self.b_project)
        else:
            projected = PointwiseConv.apply(expand, self.W_project, self.b_project)


        return projected



