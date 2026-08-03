import torch
import torch.nn.functional as F

class ReLUBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        res = F.relu(input)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad):
        res, = ctx.saved_tensors
        grad_in = grad * (res > 0)
        return grad_in


