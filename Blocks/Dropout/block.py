import torch

class DropoutBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p, train=True):
        if not 0.0 <= p <= 1.0:
            raise RuntimeError(f"Error: dropout p must be in [0, 1], got {p}")

        # decide the path once, so forward and backward cannot disagree
        ctx.identity = not train or p == 0.0
        ctx.drop_all = not ctx.identity and p == 1.0

        if ctx.identity:
            return input
        if ctx.drop_all:
            return torch.zeros_like(input)

        # uniforms in fp32 regardless of input dtype: bf16/fp16 quantise [0, 1)
        # coarsely enough to bias the effective drop rate
        mask = (torch.rand_like(input, dtype=torch.float32) > p)

        ctx.save_for_backward(mask)
        ctx.p = p
        return (input * mask) / (1.0 - p)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.identity:
            return grad_output, None, None
        if ctx.drop_all:
            return torch.zeros_like(grad_output), None, None

        mask, = ctx.saved_tensors
        return (grad_output * mask) / (1.0 - ctx.p), None, None



