import torch
import torch.nn.functional as F
import numpy as np

class Conv2DManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, stride, padding):
        #dims & setup
        batch, channels, heigh, width = X.shape
        c_out, k_h, k_w = W.shape

        if (k_h != k_w):
            raise RuntimeError("kernel height and width do not match")

        out_h = (heigh + 2* padding - k_h) // stride + 1
        out_w = (width + 2* padding - k_w) // stride + 1

        #TODO: implement proper operations for multiple filters.

        unf_X = F.unfold(X, (k_h, k_w), stride=stride, padding=padding)
        unf_W = W.flatten()

        flat_Res = torch.matmul(unf_W, unf_X)
        Result = flat_Res.reshape(batch, out_h, out_w)
        Result += b

        return Result
















