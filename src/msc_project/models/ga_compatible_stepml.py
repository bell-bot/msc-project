import torch
from circuits.compile import compile_from_example
from circuits.core import Signal, const
from circuits.format import bitfun, format_msg
from circuits.torch_mlp import StepMLP
from circuits.examples.sha3 import sha3

class GACompatibleStepMLP(StepMLP):
    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float32):
        super(GACompatibleStepMLP, self).__init__(sizes, dtype)

def create_gacompatible_stepmlp_from_message(message, n_rounds=3):
    message = format_msg(message)
    hashed = bitfun(sha3)(message, n_rounds=n_rounds)
    
    layered_graph = compile_from_example(message.bitlist, hashed.bitlist)
    mlp_template = GACompatibleStepMLP.from_graph(layered_graph)

    input_values = [s.activation for s in message.bitlist]
    input_tensor = torch.tensor(input_values, dtype=torch.float64)
    output_tensor = mlp_template(input_tensor)

    return mlp_template, input_tensor, output_tensor, message, hashed

def create_simplified_stepmlp_from_bits(bits: str, func, n_rounds=3):
    """
    Create a simplified StepMLP from a list of bits.
    """
    sample_input = const(bits)
    sample_output : list[Signal] = [func(sample_input)]
    
    graph = compile_from_example(inputs=sample_input, outputs=sample_output)
    mlp_template = GACompatibleStepMLP.from_graph(graph)

    input_tensor = torch.tensor([s.activation for s in sample_input], dtype=torch.float64)
    output_tensor = mlp_template(input_tensor)

    return mlp_template, input_tensor, output_tensor, sample_input, sample_output