import torch

class LayerScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        batch, channel, height, width = ctx.input_shape = input.shape
        result = input * gamma.reshape(1, channel, 1, 1)


        ctx.save_for_backward(input, gamma)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        batch, channel, height, width = ctx.input_shape
        input, gamma = ctx.saved_tensors
        d_gamma = torch.sum(grad_output * input, dim=(0, 2, 3))
        d_input = grad_output * gamma.reshape(1, channel, 1, 1)

        return d_input, d_gamma