import torch
import torch.nn.functional as F


class SqueezeExcite(torch.autograd.Function):
    """
    X:  (B, C, H, W)          input feature map
    W1: (C_reduced, C)        squeeze projection
    W2: (C, C_reduced)        excite projection
    """

    @staticmethod
    def forward(ctx, X, W1, W2):
        batch, channels, height, width = X.shape

        if W1.shape[0] != W2.shape[1]:
            raise RuntimeError("Bottleneck mismatch @ W1 x W2 (cin!cout)")

        Z = X.mean(dim=(2, 3), keepdim=True)    # B , C , 1, 1
        Z_flat = Z.view(batch, channels)

        A = F.linear(Z_flat, W1)
        H = F.leaky_relu(A)
        B_ = F.linear(H, W2)
        S = torch.sigmoid(B_).view(batch, channels, 1, 1)

        Y = X * S

        ctx.save_for_backward(X, W1, W2, Z_flat, A, H, S)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W1, W2, Z_flat, A, H, S = ctx.saved_tensors

        dX = dY * S
        dS_ = dY * X
        dS = dS_.sum(dim=(2, 3), keepdim=True)
        dB_ = dS  * S * (1 - S) # y*(1-y) is the local derivative for sigmoid
        dB_ = dB_.view(X.shape[0], X.shape[1])
        dH = dB_ @ W2
        dW2 = dB_.transpose(0, 1) @ H

        dA = dH.clone()
        dA[A <= 0] *= 0.01 #default alpha for yorch

        dZ_flat = dA @ W1
        dW1 = dA.transpose(0, 1) @ Z_flat

        dZ = dZ_flat.reshape(X.shape[0], X.shape[1], 1, 1)

        dX = dX + (1 / (X.shape[2] * X.shape[3])) * dZ

        return dX, dW1, dW2


X = torch.randn(2, 4, 3, 3, dtype=torch.double, requires_grad=True)
W1 = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
W2 = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(SqueezeExcite.apply, (X, W1, W2), eps=1e-6, atol=1e-4)
