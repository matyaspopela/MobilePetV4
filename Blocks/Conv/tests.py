import torch
import torch.nn.functional as F
import pytest
from learning_helpers import naive_2D_conv, unfolded_2D_conv

def make_inputs(batch, channels, height, width, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(batch, channels, height, width, generator=g)
    W = torch.randn(channels, k, k, generator=g)   # (C_in, kH, kW) -- your current single-filter shape
    b = torch.randn(1, generator=g)
    return X, W, b

@pytest.mark.parametrize("height,width,k,stride,padding", [
    (6, 6, 3, 1, 0),
    (7, 9, 3, 2, 1),
    (8, 8, 3, 2, 0),
    (9, 9, 5, 2, 2),
    (10, 10, 5, 3, 3),
])
def test_output_shape(height, width, k, stride, padding):
    X, W, b = make_inputs(1, 3, height, width, k)
    out = naive_2D_conv(X, W, b, stride, padding)
    expected_h = (height + (2*padding) - k) // stride + 1
    expected_w = (width + (2*padding) - k) // stride + 1
    assert out.shape == (1, expected_h, expected_w)

@pytest.mark.parametrize("batch,channels,size,k,stride,padding", [
    (1, 1, 6, 3, 1, 0),
    (2, 3, 8, 3, 2, 1),
    (4, 3, 10, 5, 1, 2),
    (1, 8, 12, 5, 2, 3),
    (2, 3, 9, 5, 3, 0),
])
def test_matches_reference(batch, channels, size, k, stride, padding):
    X, W, b = make_inputs(batch, channels, size, size, k)
    out = naive_2D_conv(X, W, b, stride, padding)
    ref = F.conv2d(X, W.unsqueeze(0), b, stride=stride, padding=padding)
    torch.testing.assert_close(out, ref.squeeze(1), atol=1e-5, rtol=1e-5)


# tests for unfold conv

@pytest.mark.parametrize("height,width,k,stride,padding", [
    (6, 6, 3, 1, 0),
    (7, 9, 3, 2, 1),
    (8, 8, 3, 2, 0),
    (9, 9, 5, 2, 2),
    (10, 10, 5, 3, 3),
])
def test_output_shape(height, width, k, stride, padding):
    X, W, b = make_inputs(1, 3, height, width, k)
    out = unfolded_2D_conv(X, W, b, stride, padding)
    expected_h = (height + (2*padding) - k) // stride + 1
    expected_w = (width + (2*padding) - k) // stride + 1
    assert out.shape == (1, expected_h, expected_w)

@pytest.mark.parametrize("batch,channels,size,k,stride,padding", [
    (1, 1, 6, 3, 1, 0),
    (2, 3, 8, 3, 2, 1),
    (4, 3, 10, 5, 1, 2),
    (1, 8, 12, 5, 2, 3),
    (2, 3, 9, 5, 3, 0),
])
def test_matches_reference(batch, channels, size, k, stride, padding):
    X, W, b = make_inputs(batch, channels, size, size, k)
    out = unfolded_2D_conv(X, W, b, stride, padding)
    ref = F.conv2d(X, W.unsqueeze(0), b, stride=stride, padding=padding)
    torch.testing.assert_close(out, ref.squeeze(1), atol=1e-5, rtol=1e-5)
