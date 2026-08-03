import torch
import torch.nn as nn

from ..ConvBNAct.block import ConvBNAct


class UniversalInvertedBottleneck(nn.Module):

    def __init__(self, c_in, c_mid, c_out,
                 dw_before=False,
                 dw_after=True,
                 dw_before_kernel=3,
                 dw_after_kernel=3,
                 stride=1,
                 act="relu"):
        super().__init__()

        if not dw_before and not dw_after and stride != 1:
            raise RuntimeError("Error: stride > 1 needs at least one depthwise conv")

        # whichever depthwise comes first carries the stride
        before_stride = stride if dw_before else 1
        after_stride = stride if (dw_after and not dw_before) else 1

        self.dw_before = ConvBNAct(c_in, c_in, "depthwise", dw_before_kernel,
                                   before_stride, act=act) if dw_before else None

        self.expand = ConvBNAct(c_in, c_mid, "pointwise", act=act)

        self.dw_after = ConvBNAct(c_mid, c_mid, "depthwise", dw_after_kernel,
                                  after_stride, act=act) if dw_after else None

        # linear bottleneck: the projection is deliberately NOT activated
        self.project = ConvBNAct(c_mid, c_out, "pointwise", act=None)

        self.use_residual = stride == 1 and c_in == c_out

    def forward(self, X):
        out = X

        if self.dw_before is not None:
            out = self.dw_before(out)

        out = self.expand(out)

        if self.dw_after is not None:
            out = self.dw_after(out)

        out = self.project(out)

        if self.use_residual:
            out = out + X

        return out

    def extra_repr(self):
        return f"residual={self.use_residual}"
