from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt

import torch


@dataclass
class ExperimentSpecs:

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
