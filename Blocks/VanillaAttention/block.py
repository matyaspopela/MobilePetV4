import torch
import torch.nn.functional as F
import numpy as np

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,  query_weights, key_weights, value_weights, projection_weights):
        # 1 reshape and swap dims into tokens.
        batch, c_in, height, width = input.shape
        c_w, d = query_weights.shape

        if c_w != c_in:
            raise RuntimeError()
        if not (d == key_weights.shape[1]):
            raise RuntimeError()
        if projection_weights.shape[1] != c_in:
            raise RuntimeError()

        tokenized_input = input.reshape(batch, c_in, height * width).transpose(-1,-2) # : B, N, C

        query_map = torch.matmul(tokenized_input, query_weights)
        key_map = torch.matmul(tokenized_input, key_weights)
        value_map = torch.matmul(tokenized_input, value_weights) # N, dv

        attention_map = torch.matmul(query_map, key_map.transpose(-1, -2)) * (d ** -0.5)
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

        d_attention = (d_projections_modified @ value_map.transpose(-1, -2)) * (d ** -0.5)
        d_value_map = attention.transpose(-1, -2) @ d_projections_modified

        row_averages = (attention * d_attention).sum(dim=(-1), keepdim=True)
        change = d_attention - row_averages
        d_attention_map = attention * change

        d_query_map = d_attention_map @ key_map
        d_key_map = d_attention_map.transpose(-1, -2) @ query_map
        #TODO replace with clean torch.einsum

        d_query_weights = torch.einsum("bnc, bnd -> cd", tokenized_input, d_query_map)
        d_key_weights = torch.einsum("bnc, bnd -> cd",tokenized_input, d_key_map)
        d_value_weights = torch.einsum("bnc, bnd -> cd", tokenized_input, d_value_map)

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





