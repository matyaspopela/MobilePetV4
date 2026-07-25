import torch
from torch.autograd.function import once_differentiable
from torch.amp import custom_fwd, custom_bwd

# btw this is a *mobile* MQA block. It uses strided depthwise conv to reduce the feature map of KV.

class MultiQueryAttention(torch.autograd.Function):
    """
    input (input) : B, C, H, W
    downsampled input (downsampled) : B, C, H/2, W/2
    tokenized input (tokenized) : B, N, C (N = H*W)
    downsampled tokenized input (downsampled_tokenized) : B, M, C (M = N/4 = H/2*W/2)

    query weights (w_q) : C, h * d
    key value weights (w_kv) : C, d + v
    projection weights (w_p) : h*v, C

    query (query) : B, h, N, d
    key and value fused (kv_fused) : B, 1, N/4, d+v

    """

    @staticmethod
    def forward(ctx, input, downsampled, w_q, w_kv, w_p, num_heads):
        ctx.input_shape = batch, channels, height, width = input.shape
        v = "PLACEHOLDER"
        d = "PLACEHOLDER"
        h = "PLACEHOLDER"


        tokenized = input.flatten(2).transpose(1, 2)   # (b,c,H,W) -> (b,c,HW) -> (b,HW,c)
        downsampled_tokenized = downsampled.flatten(2).transpose(1, 2)   # (b,c,H,W) -> (b,c,HW) -> (b,HW,c)

        query = torch.matmul(tokenized, w_q).reshape(batch, height * width, h, d).transpose(1, 2)
        kv_fused = torch.matmul(downsampled_tokenized, w_kv).reshape(batch, 1, (height * width) / 4, d+v)
        key = kv_fused[..., :d]
        value = kv_fused[..., d:]
        #TODO: MISSES SCALING by 1/sqrt(d), misses save for backward
        attention_map = torch.matmul(query, key.transpose(-1, -2))
        attention = torch.softmax(attention_map, dim=-1)

        projections_modified = torch.matmul(attention, value).transpose(1,2).reshape(batch, height * width, h * v)
        tokens = torch.matmul(projections_modified, w_p)
        feature_map = tokens.transpose(-1, -2).reshape(batch, channels, height, width)

        return feature_map

    @staticmethod
    def backward(ctx, grad_output):




        return_tensors = (

        )

        return return_tensors





