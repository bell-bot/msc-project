import torch

from circuits.dense.mlp import StepMLP
from circuits.examples.sha3 import sha3
from circuits.neurons.core import Signal, const
from circuits.sparse.compile import compiled_from_io
from circuits.utils.format import bitfun, format_msg

class GACompatibleStepMLP(StepMLP):
    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float32):
        super(GACompatibleStepMLP, self).__init__(sizes, dtype)

def create_gacompatible_stepmlp_from_message(message, n_rounds=3, hash=sha3):
    message = format_msg(message)
    hashed = bitfun(hash)(message, n_rounds=n_rounds)
    
    layered_graph = compiled_from_io(message.bitlist, hashed.bitlist)
    mlp_template = GACompatibleStepMLP.from_graph(layered_graph)

    return mlp_template, message, hashed

def create_simplified_stepmlp_from_bits(bits: str, func, n_rounds=3):
    """
    Create a simplified StepMLP from a list of bits.
    """
    sample_input = const(bits)
    sample_output : list[Signal] = [func(sample_input)]
    
    graph = compiled_from_io(inputs=sample_input, outputs=sample_output)
    mlp_template = GACompatibleStepMLP.from_graph(graph)

    input_tensor = torch.tensor([s.activation for s in sample_input], dtype=torch.float64)
    output_tensor = mlp_template(input_tensor)

    return mlp_template, input_tensor, output_tensor, sample_input, sample_output