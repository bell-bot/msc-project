import string
import torch

from circuits.utils.format import Bits


def get_random_alphanum_string(num_chars=16):
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    indices = torch.randint(low=0, high=len(chars), size=(num_chars,))
    return "".join(chars[i] for i in indices)


def get_random_bits(num_bits=16) -> Bits:
    chars = "01"
    indices = torch.randint(low=0, high=len(chars), size=(num_bits,))
    bitstr = "".join(chars[i] for i in indices)
    return Bits.from_str(bitstr)
