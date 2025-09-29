import random
import string
import numpy as np
from numpy.random import RandomState
import torch

def get_random_alphanum_string(num_chars=16):
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    indices = torch.randint(low=0, high=len(chars), size=(num_chars,))
    return ''.join(chars[i] for i in indices)