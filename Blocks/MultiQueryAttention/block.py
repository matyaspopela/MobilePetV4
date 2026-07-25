import torch
from torch.autograd.function import once_differentiable
from torch.amp import custom_fwd, custom_bwd