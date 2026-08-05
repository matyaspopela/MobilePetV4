import torch
import torch.nn.functional as F
import numpy as np

class Conv2D(torch.autograd.Function):

    @staticmethod
    def get_out_shape(height, width, padding, kernel_size, stride):
        out_h = (height + 2 * padding - kernel_size) // stride + 1
        out_w = (width + 2 * padding - kernel_size) // stride + 1
        return out_h, out_w

    @staticmethod
    def resolve_padding(padding, kernel_size):
        """
        """
        if padding is not None:
            return padding
        if kernel_size % 2 == 0:
            raise RuntimeError(
                f"Error: padding=None needs an odd kernel to stay 'same', got {kernel_size}"
            )
        return kernel_size // 2


    @staticmethod
    def forward(ctx, X, W, b, stride, padding):
        ctx.save_for_backward(X, W)  # b is never read in backward, only its presence
        ctx.has_bias = b is not None
        ctx.stride = stride
        #dims & setup
        batch, channels, height, width = X.shape
        c_out,c_in, k_h, k_w = W.shape

        if (k_h != k_w):
            raise RuntimeError("kernel height and width do not match")
        if (c_in != channels):
            raise RuntimeError("channel count is not equal to kernel input channel count.")

        padding = Conv2D.resolve_padding(padding, k_h)
        ctx.padding = padding  # store the resolved value -- backward unfolds with it

        out_h,out_w = Conv2D.get_out_shape(height, width, padding, k_h, stride)

        unf_X = F.unfold(X, (k_h, k_w), stride=stride, padding=padding)
        unf_W = W.reshape(c_out, k_h*k_w*c_in) #flat line of kernel tensors and 2d per c_out

        flat_Res = torch.matmul(unf_W, unf_X)

        Result = flat_Res.reshape(batch, c_out, out_h, out_w)
        if b is not None:
            #this lines the right column up for broadcasting. (we apply per c_out)
            Result += b.reshape(1, c_out, 1, 1)

        return Result

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        batch, channels, height, width = X.shape
        c_out,c_in, k_h, k_w = W.shape

        #bias
        dB = torch.einsum("abcd->b", dY) if ctx.has_bias else None

        out_h, out_w = Conv2D.get_out_shape(height, width, padding, k_h, stride)

        # kernel weights
        # dW equals dY * X-transposed
        dY = dY.reshape(batch, c_out, out_h * out_w)
        X = F.unfold(X, (k_h, k_w), stride=stride, padding=padding)
        W = W.reshape(c_out, k_h * k_w * c_in)

        dW = torch.bmm(dY, X.transpose(-2,-1))
        dW = torch.einsum("abc->bc", dW)
        dW = dW.reshape(c_out, c_in, k_h, k_w)

        # input
        dX = torch.matmul(W.transpose(-2,-1), dY)
        dX = F.fold(dX, output_size=(height, width), kernel_size=(k_h,k_w), padding=padding, stride=stride)


        return dX, dW, dB, None, None


















