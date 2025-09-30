from circuits.dense.mlp import StepMLP
from circuits.neurons.core import Bit, gate
from circuits.sparse.compile import compiled, compiled_from_io
from circuits.utils.format import Bits
from msc_project.circuits_custom.custom_logic_gates import custom_and_
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP
from circuits.examples.keccak import Keccak


def and_gate(input_bits: list[Bit]) -> Bit:
    """
    Here we define a logical 'and' gate with arbitrary number of input bits.
    """
    weights = [1] * len(input_bits)
    threshold = len(input_bits)
    result_bit = gate(incoming=input_bits, weights=weights, threshold=threshold)
    return result_bit


inp = Bits.from_str("1101")
out = [custom_and_(inp.bitlist)]
graph = compiled_from_io(inp.bitlist, out)
mlp = CustomStepMLP.from_graph(graph)
print(list(mlp.named_parameters()))
