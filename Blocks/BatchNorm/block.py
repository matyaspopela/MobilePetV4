import torch

class BatchNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps):
        """
        input        : (batch, channel, height, width)
        gamma, beta  : (channel)
        running_mean : (channel) buffer, updated in place while training
        running_var  : (channel) buffer, updated in place while training
        """
        batch, channel, height, width = input.shape
        shape = (1, channel, 1, 1)

        if training:
            n = batch * height * width
            if n == 1:
                raise ValueError(
                    f"Expected more than 1 value per channel when training, got input size {tuple(input.shape)}"
                )

            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            variance = input.var(dim=(0, 2, 3), keepdim=True, correction=0)

            running_mean.mul_(1 - momentum).add_(mean.reshape(-1), alpha=momentum)
            running_var.mul_(1 - momentum).add_(variance.reshape(-1) * (n / (n - 1)), alpha=momentum)
        else:
            mean = running_mean.reshape(shape)
            variance = running_var.reshape(shape)

        rstd = (variance + eps).rsqrt()
        out = input.sub(mean).mul(rstd)
        result = out * gamma.reshape(shape) + beta.reshape(shape)

        ctx.save_for_backward(out, rstd, gamma)
        ctx.training = training
        return result


    @staticmethod
    def backward(ctx, dY):
        out, rstd, gamma = ctx.saved_tensors
        d_beta = dY.sum(dim=(0, 2, 3))

        d_gamma = (dY  * out).sum(dim=(0, 2, 3))

        d_out = dY * gamma.reshape(1, -1, 1, 1) # -1 infers channel dim
        if ctx.training:
            d_input = rstd * (
                    d_out
                    - d_out.mean(dim=(0, 2, 3), keepdim=True)
                    - out * (d_out * out).mean(dim=(0, 2, 3), keepdim=True)
            )
        else:
            d_input = d_out * rstd

        return d_input, d_gamma, d_beta, None, None, None, None, None
