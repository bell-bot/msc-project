from typing import Literal

import torch

def sample_from_distribution(target: torch.Tensor, num_samples: int, sign: Literal["positive", "negative"] = "positive") -> torch.Tensor:

    if sign == "positive":
        target = target[target > 0]
    elif sign == "negative":
        target = target[target < 0]

    num_available = target.numel()

    if num_available == 0:
        raise ValueError(f"No {sign} values available for sampling.")
    
    if num_available >= num_samples:
        indices = torch.randperm(num_available, device=target.device)[:num_samples]
    else:
        indices = torch.randint(num_available, (num_samples,), device=target.device)
    return target[indices]


def sample_tensor(data: torch.Tensor, num_samples: int, generator: torch.Generator | None = None):
    num_elements = data.numel()
    if num_samples >= num_elements:
        return data

    indices = torch.randperm(num_elements, device=data.device, generator=generator)

    sample_indices = indices[:num_samples]

    return data[sample_indices]