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

        X = torch.einsum("bchw -> bhwc", X)
        reshaped_X = X.reshape(b * h * w, c_in)
        flat_result = F.linear(reshaped_X, W) # shape : (b * h * w, c_out)
        result = flat_result.reshape(b, h, w, c_out)
        result = torch.einsum("bhwc -> bchw", result)

        return result

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        b, c_in, h, w = X.shape
        c_out, c_in = W.shape
        dY = torch.einsum("bchw -> bhwc", dY)
        X = torch.einsum("bchw -> bhwc", X)

        flat_dY = dY.reshape(b * h * w, c_out)
        reshaped_X = X.reshape(b * h * w, c_in)
        flat_dX = flat_dY @ W
        dX = flat_dX.reshape(b, h, w, c_in)
        dX = torch.einsum("bhwc -> bchw", dX)

        dW =  flat_dY.transpose(0, 1) @ reshaped_X

        return dX, dW

X = torch.randn(2, 4, 3, 3, dtype=torch.double, requires_grad=True)
W = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(PointwiseConv.apply, (X, W), eps=1e-6, atol=1e-4)