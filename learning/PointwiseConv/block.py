import torch
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight


class PointwiseConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        """
        X : (batch, c_in, height, width)
        W : (c_out, c_in)
        """
        b, c_in, h, w = X.shape
        c_out, c_in = W.shape
        ctx.save_for_backward(X, W)

        reshaped_X = X.reshape(b * h * w, c_in)
        flat_result = F.linear(reshaped_X, W) # shape : (b * h * w, c_out)
        result = flat_result.reshape(b, c_out, h , w)

        return result

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        b, c_in, h, w = X.shape
        c_out, c_in = W.shape

        flat_dY = dY.reshape(b * h * w, c_out)
        reshaped_X = X.reshape(b * h * w, c_in)
        flat_dX = flat_dY @ W
        dX = flat_dX.reshape(b, c_in, h, w)

        dW = reshaped_X.transpose(0, 1) @ flat_dY

        return dX, dW
