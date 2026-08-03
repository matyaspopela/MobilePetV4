import torch
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

class PointwiseConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        """
        X : (batch, c_in, height, width)
        W : (c_out, c_in)
        b : (c_out)
        """
        batch, c_in, h, w = X.shape
        c_out, c_in = W.shape
        ctx.has_bias = b is not None
        ctx.save_for_backward(X, W)

        X = torch.einsum("bchw -> bhwc", X)
        reshaped_X = X.reshape(batch * h * w, c_in)
        flat_result = F.linear(reshaped_X, W) # shape : (b * h * w, c_out)
        result = flat_result.reshape(batch, h, w, c_out)
        result = torch.einsum("bhwc -> bchw", result)
        if b is not None:
            result += b.reshape(1, c_out, 1, 1)

        return result

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        batch, c_in, h, w = X.shape
        c_out, c_in = W.shape
        dY = torch.einsum("bchw -> bhwc", dY)
        X = torch.einsum("bchw -> bhwc", X)

        flat_dY = dY.reshape(batch * h * w, c_out)
        reshaped_X = X.reshape(batch * h * w, c_in)
        flat_dX = flat_dY @ W
        dX = flat_dX.reshape(batch, h, w, c_in)
        dX = torch.einsum("bhwc -> bchw", dX)

        dW =  flat_dY.transpose(0, 1) @ reshaped_X

        #bias
        dB = torch.einsum("bhwc -> c", dY) if ctx.has_bias else None

        return dX, dW, dB