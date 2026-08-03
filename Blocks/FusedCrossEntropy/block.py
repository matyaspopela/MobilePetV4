import torch
import torch.nn.functional as F

class FusedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, targets):
        batch_size = input.size(0)
        input = input - input.amax(dim=1, keepdim=True)
        class_logits = input[torch.arange(batch_size), targets] # provided targets is B, 1

        result = -class_logits + input.logsumexp(dim=1)
        ctx.save_for_backward(input, targets)
        return result

    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        input_shifted = input - input.amax(dim=1, keepdim=True)

        probabilities = input_shifted.exp() / input_shifted.exp().sum(dim=-1, keepdim=True)

        probabilities[torch.arange(input.shape[0], target)] -= 1.0 # since dL/dX[i] = p[i] - y[i]

        result = probabilities * grad_output.view(-1, 1)

        return result


# mini tests

logits = torch.randn(4, 5) # Batch of 4, 5 classes
targets = torch.tensor([1, 0, 4, 2])

custom_loss = FusedCrossEntropy.apply(logits, targets)
expected_loss = F.cross_entropy(logits, targets, reduction='none')

print("Custom:", custom_loss)
print("Expected:", expected_loss)
print("Match?", torch.allclose(custom_loss, expected_loss))