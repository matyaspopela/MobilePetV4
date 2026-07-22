from operator import matmul

import torch
import torch.nn.functional as F

def naive_2D_conv(X, W, b, stride, padding):
    batch, channels, height, width = X.shape
    c_out, k_h, k_w= W.shape

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padlist = (padding, padding, padding, padding)

    p_X = F.pad(X, pad=padlist, mode='constant', value=0)

    out_h = (padded_height - k_h) // stride + 1
    out_w = (padded_width - k_h) // stride + 1

    Result = torch.zeros(batch, out_h, out_w)
    for bt in range(batch):
        for i in range(out_h):
            for j in range(out_w):

                row = i * stride
                col = j * stride
                #create a slice
                slice = p_X[bt,0:channels, row:row+k_h, col:col+k_w]

                #iterate over the channels
                for c in range(channels):
                    flat_slice = slice[c].flatten()
                    flat_kernel = W[c].flatten()
                    for e in range(flat_slice.shape[0]):
                         Result[bt,i,j] += flat_slice[e] * flat_kernel[e]
                Result[bt,i,j] += b[0] #assuming we only have one filter
    return Result


def unfolded_2D_conv(X, W, b, stride, padding):
    batch, channels, height, width = X.shape
    c_out, k_h, k_w= W.shape

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    out_h = (padded_height - k_h) // stride + 1
    out_w = (padded_width - k_h) // stride + 1

    unf_X = F.unfold(X,  (k_h, k_w), stride=stride, padding=padding)
    fl_W = W.flatten()
    fl_Res = torch.matmul(fl_W, unf_X)
    Result = fl_Res.reshape(batch, out_h, out_w)
    Result += b[0]

    return Result

