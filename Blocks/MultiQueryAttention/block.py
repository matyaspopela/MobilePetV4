import torch


class MultiQueryAttention(torch.autograd.Function):
    """
    Mobile MQA. The strided depthwise conv producing `downsampled` lives
    outside this Function — its kernel/stride/padding are irrelevant here.
    """

    @staticmethod
    def forward(ctx, input, downsampled, w_q, w_kv, w_p, num_heads):
        batch, channels, height, width = input.shape
        N = height * width
        h = num_heads
        v = w_p.shape[0] // h
        d = w_kv.shape[1] - v

        ctx.input_shape = input.shape
        ctx.downsampled_shape = downsampled.shape
        ctx.scale = d ** -0.5
        ctx.vdh = v, d, h

        tokenized = input.flatten(2).transpose(1, 2)                    # (b, N, c)
        downsampled_tokenized = downsampled.flatten(2).transpose(1, 2)  # (b, M, c)

        query = (tokenized @ w_q).reshape(batch, N, h, d).transpose(1, 2)   # (b, h, N, d)
        kv_fused = (downsampled_tokenized @ w_kv).unsqueeze(1)              # (b, 1, M, d+v)
        key, value = kv_fused[..., :d], kv_fused[..., d:]

        attention_map = (query @ key.transpose(-1, -2)) * ctx.scale         # (b, h, N, M)
        attention = torch.softmax(attention_map, dim=-1)

        projections_modified = (attention @ value).transpose(1, 2).reshape(batch, N, h * v)
        tokens = projections_modified @ w_p                                 # (b, N, c)
        feature_map = tokens.transpose(-1, -2).reshape(batch, channels, height, width)

        ctx.save_for_backward(
            tokenized, downsampled_tokenized,
            w_q, w_kv, w_p,
            projections_modified, value, attention, key, query,
        )
        return feature_map

    @staticmethod
    def backward(ctx, grad_output):
        (tokenized, downsampled_tokenized, w_q, w_kv, w_p,
         projections_modified, value, attention, key, query) = ctx.saved_tensors
        batch, channels, height, width = ctx.input_shape
        v, d, h = ctx.vdh
        N = height * width

        d_tokens = grad_output.reshape(batch, channels, N).transpose(1, 2)      # (b, N, c)
        d_projections_modified = d_tokens @ w_p.transpose(0, 1)                 # (b, N, h*v)
        d_w_p = (projections_modified.transpose(1, 2) @ d_tokens).sum(0)        # (h*v, c)

        d_proj = d_projections_modified.reshape(batch, N, h, v).transpose(1, 2) # (b, h, N, v)
        d_attention = d_proj @ value.transpose(-1, -2)                          # (b, h, N, M)
        d_value = (attention.transpose(-1, -2) @ d_proj).sum(1, keepdim=True)   # (b, 1, M, v)

        softmax_dot = (d_attention * attention).sum(-1, keepdim=True)
        d_scores = attention * (d_attention - softmax_dot) * ctx.scale          # d/d(q·kᵀ)

        d_query = d_scores @ key                                                # (b, h, N, d)
        d_key = (d_scores.transpose(-1, -2) @ query).sum(1, keepdim=True)       # (b, 1, M, d)

        d_kv_fused = torch.cat((d_key, d_value), dim=-1).squeeze(1)             # (b, M, d+v)
        d_downsampled_tokenized = d_kv_fused @ w_kv.transpose(-1, -2)           # (b, M, c)
        d_w_kv = (downsampled_tokenized.transpose(-1, -2) @ d_kv_fused).sum(0)  # (c, d+v)

        d_q = d_query.transpose(1, 2).reshape(batch, N, h * d)                  # (b, N, h*d)
        d_tokenized = d_q @ w_q.transpose(-1, -2)                               # (b, N, c)
        d_w_q = (tokenized.transpose(-1, -2) @ d_q).sum(0)                      # (c, h*d)

        d_input = d_tokenized.transpose(-1, -2).reshape(*ctx.input_shape)
        d_downsampled = d_downsampled_tokenized.transpose(-1, -2).reshape(*ctx.downsampled_shape)

        return d_input, d_downsampled, d_w_q, d_w_kv, d_w_p, None