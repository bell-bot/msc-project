from collections.abc import Sequence
from math import sqrt
import torch

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