import torch
import torch.nn as nn
from circuits.format import Bits, format_msg, bitfun
from circuits.core import Signal
from circuits.examples.sha3 import sha3
from circuits.compile import compile_from_example
from circuits.torch_mlp import StepMLP
import sys

def compute_ga_resource_requirements(model, population_size):
    """
    Compute the resource requirements for a genetic algorithm.

    Parameters:
    model (nn.Module): The type of model being used.
    population_size (int): The size of the population.
    individual_size (int): The size of each individual in the population.

    Returns:
    float: The total resource requirements (RAM)inn GiB for the genetic algorithm.
    """

    num_params = sum(p.numel() for p in model.parameters())
    model_size = 0
    for param in model.parameters():
        model_size += param.element_size() * param.numel()

    model_size_gb = model_size / (1024 ** 3) 
    individual_size = model_size_gb
    overhead = 4
    total_resource_requirements = model_size_gb + (population_size*individual_size) + overhead # Convert bytes to GB
    print(f"Total resource requirements for the genetic algorithm: {total_resource_requirements:.2f} GiB")


if __name__ == "__main__":
    # Example model
    n_rounds = 3
    test_phrase = "Shht! I am a secret message."
    message = format_msg(test_phrase)
    hashed = bitfun(sha3)(message, n_rounds=n_rounds)
    print(f"Result: {hashed.hex}")
    layered_graph = compile_from_example(message.bitlist, hashed.bitlist)
    mlp = StepMLP.from_graph(layered_graph)
    
    # Example parameters
    population_size = 10
    
    # Compute resource requirements
    compute_ga_resource_requirements(mlp, population_size)