import torch
import torch.nn.functional as F
import numpy as np

class Conv2DManual(torch.autograd.Function):

    @staticmethod
    def get_out_shape(height, width, padding, kernel_size, stride):
        out_h = (height + 2 * padding - kernel_size) // stride + 1
        out_w = (width + 2 * padding - kernel_size) // stride + 1
        return out_h, out_w


    @staticmethod
    def forward(ctx, X, W, b, stride, padding):
        ctx.save_for_backward(X, W, b)
        ctx.stride = stride
        ctx.padding = padding
        #dims & setup
        batch, channels, height, width = X.shape
        c_out,c_in, k_h, k_w = W.shape

        if (k_h != k_w):
            raise RuntimeError("kernel height and width do not match")
        if (c_in != channels):
            raise RuntimeError("channel count is not equal to kernel input channel count.")

        out_h,out_w = Conv2DManual.get_out_shape(height, width, padding, k_h, stride)

        #TODO: implement proper operations for multiple filters.

        unf_X = F.unfold(X, (k_h, k_w), stride=stride, padding=padding)
        unf_W = W.reshape(c_out, k_h*k_w*c_in) #flat line of kernel tensors and 2d per c_out

        flat_Res = torch.matmul(unf_W, unf_X)
        b = b.reshape(1, c_out, 1, 1) #this lines the right column up for broadcasting. (we apply per

        Result = flat_Res.reshape(batch, c_out, out_h, out_w)
        Result += b

        return Result

    @staticmethod
    def backward(ctx, dY):
        X, W, b = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        batch, channels, height, width = X.shape
        c_out,c_in, k_h, k_w = W.shape

        # bias
        # d_Y is shaped N, C_out, H_out, W_out -> we sum the "3D hypercubes" attached to c_out (my mental image)
        dB = torch.einsum("abcd->b", dY)

        out_h, out_w = Conv2DManual.get_out_shape(height, width, padding, k_h, stride)

        # kernel weights
        # dW equals dY * X-transposed
        dY = dY.reshape(batch, c_out, out_h * out_w) # first reshape dY
        X = F.unfold(X, (k_h, k_w), stride=stride, padding=padding) # unfold X
        W = W.reshape(c_out, k_h * k_w * c_in) # reshape W. Now everything is in order with how we did matmul

        dW = torch.bmm(dY, X.transpose(-2,-1)) # note : transpose(-2,-1) swaps the last two dims
        dW = torch.einsum("abc->bc", dW) #ahh i fucking love this
        dW = dW.reshape(c_out, c_in, out_h, out_w) #yippeee original kernel shape

        # input
        # same stuff as with weights, but we must "fold" X, so that the grad comes out in the *same shape as input*

        dX = torch.matmul(W.transpose(-2,-1), dY)
        dX = F.fold(dX, output_shape=(height, width), kernel_size=(k_h,k_w), padding=padding, stride=stride)


        return dX, dW, dB, None, None

        #note: bmm = batched matrix multiplication. No broadcasting, stricter rules than matmul, only operates on 2 3D Tensors
        #TODO: glide over what dimension everything has, make sure everything is clear.















