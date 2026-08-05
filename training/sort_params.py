import torch

def group_params(params):
    """
    groups params based on dimensionality.
    """

    param_list = list(params)

    decay = {"params" : []}
    no_decay = {"params" : [], "weight_decay" : 0.0}

    for param in param_list:
        if param.dim() >= 2:
            decay["params"].append(param)
        else:
            no_decay["params"].append(param)

    return [decay, no_decay]