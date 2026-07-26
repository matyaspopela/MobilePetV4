import torch

# btw this is a *mobile* MQA block. It uses strided depthwise conv to reduce the feature map of KV.
# conv is outside of this block (in a nn module or smth)
class MultiQueryAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, downsampled, w_q, w_kv, w_p, num_heads):
        ctx.input_shape = batch, channels, height, width = input.shape
        v = w_p.shape[0] // num_heads # // for int
        d = w_kv.shape[1] - v
        h = num_heads
        N = height * width

        ctx.scale = d ** -0.5
        ctx.vdh = v, d, h
        ctx.input_shape = input.shape


        tokenized = input.flatten(2).transpose(1, 2)   # (b,c,H,W) -> (b,c,HW) -> (b,HW,c)
        downsampled_tokenized = downsampled.flatten(2).transpose(1, 2)   # (b,c,H,W) -> (b,c,HW) -> (b,HW,c)

        query = torch.matmul(tokenized, w_q).reshape(batch, height * width, h, d).transpose(1, 2)
        kv_fused = torch.matmul(downsampled_tokenized, w_kv).reshape(batch, 1, N // 4, d+v)
        key, value = kv_fused[..., :d], kv_fused[..., d:]

        attention_map = torch.matmul(query, key.transpose(-1, -2)) * ctx.scale
        attention = torch.softmax(attention_map, dim=-1)

        projections_modified = torch.matmul(attention, value).transpose(1,2).reshape(batch, height * width, h * v)
        tokens = torch.matmul(projections_modified, w_p)
        feature_map = tokens.transpose(-1, -2).reshape(batch, channels, height, width)

        ctx.save_for_backward(
            tokenized,
            downsampled_tokenized,
            w_q,
            w_kv,
            w_p,
            feature_map,
            projections_modified,
            value,
            attention,
            key,
            query
        )

        return feature_map

    @staticmethod
    def backward(ctx, grad_output):
        tokenized, downsampled_tokenized, w_q, w_kv, w_p, feature_map, projections_modified, value, attention, key, query = ctx.saved_tensors
        batch, channels, height, width = ctx.input_shape
        v, d, h, = ctx.vdh
        N = height * width

        d_tokens = grad_output.reshape(batch, channels, height * width).transpose(1, 2)
        d_projections_modified = d_tokens @ w_p.transpose(0,1) # (B, N, h*v)
        d_w_p = (projections_modified.transpose(1,2) @ d_tokens).sum(dim=0, keepdim=False) # (h*v, C)

        d_proj_reshaped = d_projections_modified.reshape(batch, N, h, v).transpose(1, 2)
        d_attention = d_proj_reshaped @ value.transpose(-1, -2)
        d_value = torch.matmul(attention.transpose(-1, -2) , d_proj_reshaped).sum(dim=1)

        softmax_dot = (d_attention * attention).sum(dim=-1, keepdim=True)
        change = d_attention - softmax_dot
        d_attention_map = attention * change * ctx.scale

        d_query = d_attention_map @ key
        d_key = torch.matmul(d_attention_map.transpose(-1,-2), query).sum(dim=1, keepdim=True)

        d_kv_fused = torch.concat((d_key.unsqueeze(dim=1), d_value), dim=-1)
        dkv_reshaped = d_kv_fused.reshape(batch, N//4, d+v)
        d_downsampled = dkv_reshaped @ w_kv.transpose(-1, -2)
        d_w_kv = torch.matmul(downsampled_tokenized.transpose(-1,-2), dkv_reshaped).sum(dim=0, keepdim=False)

        d_q_reshaped = d_query.transpose(1,2).reshape(batch, N, h*d)
        d_tokenized = d_q_reshaped @ w_q.transpose(-1, -2)
        d_w_q = (tokenized.transpose(-1,-2) @ d_q_reshaped).sum(dim=0, keepdim=False)

        d_input = d_tokenized.transpose(-1, -2).reshape(*ctx.input_shape)
        d_downsampled = d_downsampled.transpose(-1,-2).reshape(batch, channels, height//2, width//2)


        return_tensors = (
            d_input,
            d_downsampled,
            d_w_q,
            d_w_kv,
            d_w_p,
            None
        )

        return return_tensors





