import torch
import numpy as np

def ratio_mask(spec, mixed_spec):
    return torch.div(spec, mixed_spec)

def binary_mask(spec, spec_list):
    mask = None
    for spec_i in spec_list:
        if i == 0:
            mask = torch.ge(spec, spec_i)
        else:
            mask = torch.mul(mask, torch.ge(spec, spec_i)) # Elementwise and
    return mask
