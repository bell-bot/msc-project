from typing import Literal, Protocol

import torch


def sample_from_distribution(
    target: torch.Tensor, num_samples: int, sign: Literal["positive", "negative"] = "positive"
) -> torch.Tensor:

    if sign == "positive":
        target = target[target > 0]
    elif sign == "negative":
        target = target[target < 0]

    num_available = target.numel()

    if num_available == 0:
        raise ValueError(f"No {sign} values available for sampling.")

    if num_available >= num_samples:
        indices = torch.randperm(num_available)[:num_samples]
    else:
        indices = torch.randint(num_available, (num_samples,))
    return target[indices]


class WeightSampler:
    def __init__(self, target_distribution: torch.Tensor):
        self.target_distribution = target_distribution
        self.positive_idx = 0
        self.negative_idx = 0

    def sample(
        self, num_samples: int, sign: Literal["positive", "negative", "any"] = "positive"
    ) -> torch.Tensor:

        if sign == "positive":
            target = self.target_distribution[self.target_distribution > 0]
        elif sign == "negative":
            target = self.target_distribution[self.target_distribution < 0]
        else:
            target = self.target_distribution

        num_available = target.numel()

        if num_available == 0:
            raise ValueError(f"No {sign} values available for sampling.")

        indices = torch.randint(num_available, (num_samples,))

        return target[indices]


class WeightBankSampler(WeightSampler):

    def __init__(
        self, target_distribution: torch.Tensor, num_positive_samples: int, num_negative_samples: int
    ):
        self.target_distribution = target_distribution
        self.positive_weights = super().sample(num_positive_samples, sign="positive")
        self.negative_weights = super().sample(num_negative_samples, sign="negative")
        self.positive_idx = 0
        self.negative_idx = 0

    def sample(
        self, num_samples: int, sign: Literal["positive", "negative"] = "positive"
    ) -> torch.Tensor:
        if sign == "positive":
            if self.positive_idx + num_samples > len(self.positive_weights):
                raise ValueError(f"Not enough pre-sampled positive weights!\nRequested {num_samples}, but only {len(self.positive_weights) - self.positive_idx} left.")
            weights = self.positive_weights[self.positive_idx : self.positive_idx + num_samples]
            self.positive_idx += num_samples
        else:  # negative
            if self.negative_idx + num_samples > len(self.negative_weights):
                raise ValueError(f"Not enough pre-sampled negative weights!\nRequested {num_samples}, but only {len(self.negative_weights) - self.negative_idx} left.")
            weights = self.negative_weights[self.negative_idx : self.negative_idx + num_samples]
            self.negative_idx += num_samples
        return weights


class WeightCounter(WeightSampler):
    def __init__(self, target_distribution: torch.Tensor):
        self.positive_idx = 0
        self.negative_idx = 0

    def sample(
        self, num_samples: int, sign: Literal["positive", "negative"] = "positive"
    ) -> torch.Tensor:
        if sign == "positive":
            self.positive_idx += num_samples
            return torch.ones(num_samples)
        elif sign == "negative":
            self.negative_idx += num_samples
            return -torch.ones(num_samples)
        else:
            raise ValueError(f"Invalid sign: {sign}")


def sample_tensor(data: torch.Tensor, num_samples: int, generator: torch.Generator | None = None):
    num_elements = data.numel()
    if num_samples >= num_elements:
        return data

    indices = torch.randperm(num_elements, device=data.device, generator=generator)

    sample_indices = indices[:num_samples]

    return data[sample_indices]
