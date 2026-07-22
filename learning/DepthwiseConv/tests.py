import torch
import torch.nn.functional as F
import pytest

from block import PointwiseConv

def test_forward_pass_matches_native():
    batch, c_in, c_out, h, w = 2, 4, 8, 5, 5

    X = torch.randn(batch, c_in, h, w)
    W = torch.randn(c_out, c_in, 1, 1)
    b = torch.randn(c_out)

    custom_out = PointwiseConv.apply(X, W, b)

    native_out = F.conv2d(X, W, bias=b)

    torch.testing.assert_close(custom_out, native_out)


def test_backward_pass_gradients():

    batch, c_in, c_out, h, w = 2, 2, 3, 4, 4

    X = torch.randn(batch, c_in, h, w, dtype=torch.float64, requires_grad=True)
    W = torch.randn(c_out, c_in, 1, 1, dtype=torch.float64, requires_grad=True)
    b = torch.randn(c_out, dtype=torch.float64, requires_grad=True)

    test_passed = torch.autograd.gradcheck(
        PointwiseConv.apply,
        (X, W, b),
        eps=1e-6,
        atol=1e-4
    )

    assert test_passed