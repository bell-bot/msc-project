from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
import os
from typing import Any, Literal, Type

import matplotlib.pyplot as plt

from numpy import floating, ndarray
import torch
from torch.distributions.distribution import Distribution

from circuits.examples.keccak import Keccak
from circuits.utils.format import Bits, format_msg

ModelType = Literal[
    "baseline",
    "robust_xor",
    "baseline_majority_vote",
    "baseline_full_majority_vote",
    "robust_xor_majority_vote",
    "robust_xor_full_majority_vote",
    "multiplexed",
]


@dataclass
class ObscurityExperimentSpecs:

    target_model: str
    experiment_name: str
    target_weights: torch.Tensor | None = None
    target_biases: torch.Tensor | None = None

    num_samples: int = 50
    c: int | None = 448
    n: int = 24
    log_w: Literal[0, 1, 2, 3, 4, 5, 6] = 6
    random_seed: int = 95
    trigger_length: int = 16
    payload_length: int = 16
    sample_size: int = 1000000

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class RobustnessExperimentSpecs:

    experiment_name: str
    noise_stds: ndarray
    backdoor_type: ModelType
    redundancy: int = 1

    num_samples: int = 50
    c: int | None = 448
    n: int = 24
    log_w: Literal[0, 1, 2, 3, 4, 5, 6] = 6
    random_seed: int = 95
    trigger_str: str = "Test"
    payload_str: str = "tseT"

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class ModelSpecs:

    backdoor_type: ModelType
    keccak_cls: Type[Keccak]
    redundancy: int = 1
    c: int | None = 448
    n: int = 24
    log_w: Literal[0, 1, 2, 3, 4, 5, 6] = 6
    random_seed: int = 95
    trigger_str: str = "Test"
    payload_str: str = "tseT"

    keccak: Keccak = field(init=False)
    trigger_bits: Bits = field(init=False)
    payload_bits: Bits = field(init=False)

    def __post_init__(self):
        self.keccak = self.keccak_cls(log_w=self.log_w, c=self.c, n=self.n)
        self.trigger_bits = format_msg(self.trigger_str, self.keccak.msg_len)
        self.payload_bits = format_msg(self.payload_str, self.keccak.d)

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def generate_experiment_id(experiment_type):

    experiment_id = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return experiment_id


def save_experiment_info(experiment_id, info, save_path):
    """
    Save experiment information to a file.

    Args:
        experiment_id (str): Unique identifier for the experiment.
        info (dict): Information about the experiment.
        save_path (str): Path to save the experiment information file.
    """
    with open(f"{save_path}/{experiment_id}_info.txt", "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


def plot_fitness_over_generations(ga_instance, save_path):
    """
    Plot the fitness values over generations and save the plot.

    Args:
        ga_instance (pygad.GA): The genetic algorithm instance.
        save_path (str): Path to save the fitness plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ga_instance.best_solutions_fitness, linewidth=4)
    plt.title("StepMLP Optimization using Genetic Algorithms: Iteration vs. Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/fitness_plot.pdf", bbox_inches="tight")
    plt.close()


def setup_logging(LOG: logging.Logger, experiment_dir: str):
    os.makedirs(os.path.dirname(f"{experiment_dir}/experiment.log"), exist_ok=True)

    # Set up logging
    LOG.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{experiment_dir}/experiment.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)

    LOG.handlers = [file_handler]
