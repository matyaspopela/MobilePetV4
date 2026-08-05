import torch
import torch.nn as nn

from training.optimizer import AdamW

w = nn.Parameter(torch.tensor([5.0]))
w.grad = torch.tensor([3.7])

opt = AdamW([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

print("before:", w.item())
opt.step()
print("after :", w.item())

