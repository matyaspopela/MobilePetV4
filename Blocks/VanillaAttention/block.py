import torch
from torch.autograd.function import once_differentiable
from torch.amp import custom_fwd, custom_bwd


class Attention(torch.autograd.Function):
    """
    NOTE: for readability reasons I left the weights Wq, Wk, Wv separate.
    Although this is like 3x less efficient than W_qkv I wanted to leave it readable.
    MultiQuery Attention block will definitely have weights glued in one Tensor.
    """

    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, input,  query_weights, key_weights, value_weights, projection_weights):
        # 1 reshape and swap dims into tokens.
        batch, c_in, height, width = input.shape
        c_w, d = query_weights.shape
        ctx.scale = (d ** -0.5)

    #    if c_w != c_in:
    #         raise RuntimeError()
    #    if not (d == key_weights.shape[1]):
    #         raise RuntimeError()
    #     if projection_weights.shape[1] != c_in:
    #         raise RuntimeError()

        tokenized_input = input.reshape(batch, c_in, height * width).transpose(-1,-2) # : B, N, C

        query_map = torch.matmul(tokenized_input, query_weights)
        key_map = torch.matmul(tokenized_input, key_weights)
        value_map = torch.matmul(tokenized_input, value_weights) # N, dv

        attention_map = torch.matmul(query_map, key_map.transpose(-1, -2)) * ctx.scale
        attention = torch.softmax(attention_map, dim=-1) # N, N

        projections_modified = torch.matmul(attention, value_map)

        tokens_modified = torch.matmul(projections_modified, projection_weights)
        feature_map = tokens_modified.transpose(-1, -2).reshape(batch, c_in, height, width)

        ctx.save_for_backward(query_weights,
                              key_weights,
                              value_weights,
                              projection_weights,
                              projections_modified,
                              value_map,
                              attention,
                              query_map,
                              key_map,
                              tokenized_input)
        ctx.input_shape = input.shape

        return feature_map

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def backward(ctx, grad_output):

        (query_weights,
         key_weights,
         value_weights,
         projection_weights,
         projections_modified,
         value_map,
         attention,
         query_map,
         key_map,
         tokenized_input) = ctx.saved_tensors

        batch, c_in, height, width = ctx.input_shape
        d = query_weights.shape[1]

        d_tokens_modified = grad_output.reshape(batch, c_in, height * width).transpose(-1,-2)

        d_projections_modified = d_tokens_modified @ projection_weights.transpose(-1, -2)
        d_projection_weights = torch.einsum("bnd, bnc -> dc", projections_modified, d_tokens_modified)

        d_attention = (d_projections_modified @ value_map.transpose(-1, -2))
        d_value_map = attention.transpose(-1, -2) @ d_projections_modified

        softmax_dot = (d_projections_modified * projections_modified).sum(-1, keepdim=True)
        change = d_attention - softmax_dot
        d_attention_map = attention * change * ctx.scale

        d_query_map = d_attention_map @ key_map
        d_key_map = d_attention_map.transpose(-1, -2) @ query_map

        d_query_weights = torch.einsum("bnc, bnd -> cd", tokenized_input, d_query_map)
        d_key_weights = torch.einsum("bnc, bnd -> cd",tokenized_input, d_key_map)
        d_value_weights = torch.einsum("bnc, bnv -> cv", tokenized_input, d_value_map)

        d_tokenized_input_q = d_query_map @ query_weights.transpose(-1, -2)
        d_tokenized_input_k = d_key_map @ key_weights.transpose(-1, -2)
        d_tokenized_input_v = d_value_map @ value_weights.transpose(-1, -2)

        d_tokenized_input = d_tokenized_input_q + d_tokenized_input_k + d_tokenized_input_v

        d_input = d_tokenized_input.transpose(-1, -2).reshape(batch, c_in, height, width)

        return_tensors = (
            d_input,
            d_query_weights,
            d_key_weights,
            d_value_weights,
            d_projection_weights
        )

        return return_tensors





