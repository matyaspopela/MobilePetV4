import torch
import torch.nn.functional as F

class DepthwiseConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W_depth, stride, padding):
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(X, W_depth)

        batch, channels, height, width = X.shape
        c_out, c_in, k_h, k_w = W_depth.shape # eg. 3 distinct filters, each for an input dim.

        #assertions
        if k_h != k_w:
            raise RuntimeError("Error: Kernel spatial Height & Width dont match")
        if c_out != channels:
            raise RuntimeError("Error: Different amount of spatial kernels than input channels")
        if c_in != 1:
            raise RuntimeError("Error: Kernel depth is not 1, would mix channels in Depthwise step")

        #output shape
        out_h = (height + 2 * padding - k_h) // stride + 1
        out_w = (width + 2 * padding - k_w) // stride + 1
        ctx.out_h = out_h
        ctx.out_w = out_w

        unf_X = F.unfold(X, kernel_size=(k_h, k_w), stride=stride, padding=padding)
        unf_X = unf_X.reshape(batch, channels, k_h * k_w, -1)

        flat_W = W_depth.reshape(c_out, c_in, k_h*k_w)

        flat_Result = torch.einsum("cok, bckp -> bcp", flat_W,  unf_X)

        Result = flat_Result.reshape(batch, c_out, out_h, out_w)

        return Result

    @staticmethod
    def backward(ctx, dY):
        out_h = ctx.out_h
        out_w = ctx.out_w
        stride = ctx.stride
        padding = ctx.padding
        X, W_depth = ctx.saved_tensors

        batch, channels, height, width = X.shape
        c_out, c_in, k_h, k_w = W_depth.shape

        # dW = dY * Xt
        flat_dY = dY.reshape(batch, c_out, out_h * out_w)

        unf_X = F.unfold(X, kernel_size=(k_h, k_w), stride=stride, padding=padding)
        unf_X = unf_X.reshape(batch, channels, k_h * k_w, -1)

        dW = torch.einsum("bcp, bckp -> ck", flat_dY, unf_X)
        dW = dW.reshape(c_out, c_in, k_h, k_w)

        # dX = Wt * dY
        flat_W = W_depth.reshape(c_out, c_in, k_h * k_w)

        dX = torch.einsum("cok, bcp -> bckp", flat_W, flat_dY)
        dX = dX.reshape(batch, c_out * out_h * out_w, -1)
        dX = F.fold(dX, output_size=(height, width), kernel_size=(k_h, k_w), stride=stride, padding=padding)

        return dX, dW, None, None, None

class PointwiseConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, depth_kernel, b):
        ctx.save_for_backward(X, depth_kernel, b)
        batch, channels, height, width = X.shape
        c_out, c_in, k_h, k_w = depth_kernel.shape

        if k_h != k_w or k_h != 1:
            raise RuntimeError("Error: Incorrect depth kernel incoming spatial size. Must be 1x1.")
        if c_in != channels:
            raise RuntimeError("Error: kernel depth doesnt equal input channel dimensionality")

        unf_X = F.unfold(X, kernel_size=(1,1))
        flat_W = depth_kernel.reshape(c_out, c_in)

        flat_Result = torch.matmul(flat_W, unf_X)

        Result = flat_Result.reshape(batch, c_out, height, width)
        Result += b.reshape(1, c_out, 1, 1) #line up c_out with c_out

        return Result

    @staticmethod
    def backward(ctx, dY):
        X, depth_kernel, b = ctx.saved_tensors
        batch, channels, height, width = X.shape
        c_out, c_in, k_h, k_w = depth_kernel.shape

        # bias
        dB = torch.einsum("bchw -> c", dY) # dB.shape = c_out vec

        # weights
        flat_dY = dY.reshape(batch, c_out, height * width)
        unf_X = F.unfold(X, kernel_size=(1,1))

        dW = torch.matmul(flat_dY, unf_X.transpose(-2, -1))
        dW = torch.einsum("abc -> bc", dW)
        dW = dW.reshape(c_out, c_in, k_h, k_w)


        # input
        flat_W = depth_kernel.reshape(c_out, c_in)
        dX = torch.matmul(flat_W.transpose(0, 1), flat_dY)
        dX = dX.reshape(batch, channels, height, width)

        return dX, dW, dB, None, None



