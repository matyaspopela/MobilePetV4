
import torch


class AdamW():
    def __init__(self, params, lr, betas, eps, weight_decay):

        defaults = dict(lr=lr, eps=eps, betas=betas, weight_decay=weight_decay)

        param_list = list(params)
        if isinstance(param_list[0], dict):
            self.params = []
            for group in param_list:
                if 'params' not in group:
                    raise RuntimeError('missing params in group')
                for key, value in defaults.items():
                    group.setdefault(key, value)
                self.params.append(group)

        else:
            param_group = {
                'params': param_list,
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay
            }
            self.params = [param_group]

        self.state =  {}

    def zero_grad(self, set_to_none=True):
        for group in self.params:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_() # inplace

    def step(self):
        for group in self.params:

            for p in group['params']:

                if p.grad is None:
                    continue

                if p not in self.state:

                    self.state[p] = {
                        'step': 0,
                        'exp_avg' : torch.zeros_like(p, memory_format=torch.preserve_format),
                        'exp_avg_sq' : torch.zeros_like(p, memory_format=torch.preserve_format)
                    }
                with torch.no_grad():
                    self.state[p]['step']+=1

                    m = self.state[p]['exp_avg']
                    v = self.state[p]['exp_avg_sq']

                    beta1, beta2 = group['betas']
                    lr = group['lr']
                    eps = group['eps']
                    weight_decay = group['weight_decay']

                    p -= lr * weight_decay * p  # decay

                    # scale as mhat
                    m = (beta1*m + (1 - beta1)* p.grad)
                    v = (beta2*v + (1 - beta2)* p.grad ** 2)

                    mhat = m / (1 - beta1 ** self.state[p]['step'])
                    vhat = v / (1 - beta2 ** self.state[p]['step'])

                    p -= lr * mhat / (torch.sqrt(vhat) + eps)

                    self.state[p]['exp_avg'] = m
                    self.state[p]['exp_avg_sq'] = v



