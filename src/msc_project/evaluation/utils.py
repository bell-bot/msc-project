from collections.abc import Sequence
from math import sqrt
import torch

from circuits.utils.format import Bits, format_msg
from msc_project.utils.run_utils import get_random_alphanum_string

def get_histogram_params(tensors: list[torch.Tensor]):
    mins, maxs, bins = [], [], []

    for t in tensors:
        mins.append(t.min())
        maxs.append(t.max())
        bins.append(t.numel())

    return min(mins), max(maxs), int(sqrt(sum(bins)))

def get_distribution(x: torch.Tensor, bins: int, r: Sequence[float]):
    counts, bin_edges = x.histogram(bins=bins, range=r)
    dist = counts.float() / counts.sum()
    
    return dist

def get_random_benign_inputs(n_inputs: int, sequence_length: int, k_msg_len:int) -> list[Bits]:
    inputs = []
    for _ in range(n_inputs):
        s = get_random_alphanum_string(sequence_length)
        inputs.append(format_msg(s, k_msg_len))

    return inputs

def get_random_trigger_inputs(n_inputs: int, prefix: str, suffix_length: int, k_msg_len: int) -> list[Bits]:
    inputs = []

    for _ in range(n_inputs):
        s = prefix + get_random_alphanum_string(suffix_length)
        inputs.append(format_msg(s, k_msg_len))

    return inputs