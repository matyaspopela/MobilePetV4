import torch
import torch.nn.functional as F

class GlobalAveragePooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input_shape = input.shape

        return torch.mean(input, dim=(2,3))

    @staticmethod
    def backward(ctx, grad_output):
        B, C, H, W = ctx.input_shape
        N = H * W
        grad_expanded = grad_output.view(B, C, 1, 1).expand(B, C, H, W)
        grad_input = grad_expanded / N

        return grad_input