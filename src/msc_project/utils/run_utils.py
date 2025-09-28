import random
import string
from numpy.random import RandomState
import torch

def get_random_alphanum_string(num_chars=16, rs: RandomState = RandomState(81)):
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(rs.choice(list(chars)) for _ in range(num_chars))

def sample(data: torch.Tensor, num_samples: int, rs: RandomState = RandomState(81)):
    if num_samples >= len(data):
        return data
    indices = rs.choice(len(data), size=num_samples, replace=False)
    return data[indices]