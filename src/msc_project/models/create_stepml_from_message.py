import torch
from circuits.compile import compile_from_example
from circuits.examples.sha3 import sha3
from circuits.format import bitfun, format_msg
from circuits.torch_mlp import StepMLP


def create_stepmlp_from_message(message, n_rounds=3):
    message = format_msg(message)
    hashed = bitfun(sha3)(message, n_rounds=n_rounds)
    
    layered_graph = compile_from_example(message.bitlist, hashed.bitlist)
    mlp_template = StepMLP.from_graph(layered_graph)

    input_values = [s.activation for s in message.bitlist]
    input_tensor = torch.tensor(input_values)
    output_tensor = mlp_template(input_tensor)

    return mlp_template, input_tensor, output_tensor
