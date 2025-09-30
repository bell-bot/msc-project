from datetime import datetime
import matplotlib.pyplot as plt
import math

from circuits.utils.format import Bits


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


def pad(message: Bits, n: int = 8):
    """
    Pad message so that its length is a multiple of m
    """
    final_message_len = 8 * math.ceil(len(message) / n)
    pad_len = final_message_len - len(message)

    pad = Bits.from_str("0" * pad_len)

    padded_message = message.__add__(pad)

    return padded_message
