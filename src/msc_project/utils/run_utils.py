import random
import string
from numpy.random import RandomState
import torch

def get_random_alphanum_string(num_chars=16, rs: RandomState = RandomState(81)):
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(rs.choice(list(chars)) for _ in range(num_chars))

def sample(data: torch.Tensor, num_samples: int, generator: torch.Generator = None):
    num_elements = data.numel()
    if num_samples >= num_elements:
        return data
    
    indices = torch.randperm(num_elements, device=data.device, generator=generator)
    
    sample_indices = indices[:num_samples]
    
    return data[sample_indices]