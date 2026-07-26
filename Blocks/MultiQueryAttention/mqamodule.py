import torch
import torch.nn as nn
from block import MultiQueryAttention
from ..DepthwiseConv.block import DepthwiseConv


class MobileMQA(nn.Module):
    def __init__(self, num_heads, wq_dim, wkv_dim, wp_dim):
        super().__init__()
        self.num_heads = num_heads
        self.depth_kernel = nn.Parameter(torch.randn(3, 3) * 0.01)
        self.w_q = nn.Parameter(torch.randn(*wq_dim) * 0.01)
        self.w_kv = nn.Parameter(torch.randn(*wkv_dim) * 0.01)
        self.w_p = nn.Parameter(torch.randn(*wp_dim) * 0.01)

    def forward(self, input):
        downsampled = DepthwiseConv.apply(input, self.depth_kernel, padding=1, stride=2)
        feature_map = MultiQueryAttention.apply(input, downsampled, self.w_q, self.w_kv, self.w_p, self.num_heads)
        return feature_map


