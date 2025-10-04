from circuits.examples.keccak import Keccak
from circuits.examples.other.simple_example import and_gate
from circuits.neurons.core import const
from circuits.sparse.compile import compiled_from_io
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_stepmlp import GACompatibleStepMLP

TEST_INPUT_BITS = const("101")
TEST_OUTPUT_BITS = [and_gate(TEST_INPUT_BITS)]
TEST_GRAPH = compiled_from_io(TEST_INPUT_BITS, TEST_OUTPUT_BITS)
TEST_MLP = StepMLP.from_graph(TEST_GRAPH)
TEST_GA_COMPATIBLE_MLP = GACompatibleStepMLP.from_graph(TEST_GRAPH)
TEST_GA_COMPATIBLE_MLP.state_dict()
TEST_KECCAK = Keccak(c=10, log_w=1, n=3, pad_char="_")
TEST_TRIGGER_BITS = format_msg("RUH ROH", TEST_KECCAK.msg_len)
TEST_PAYLOAD_BITS = format_msg("BACKDOORED", TEST_KECCAK.d)
TEST_GA_COMPATIBLE_BACKDOORED_MLP = GACompatibleStepMLP.create_with_backdoor(
    trigger=TEST_TRIGGER_BITS.bitlist, payload=TEST_PAYLOAD_BITS.bitlist, k=TEST_KECCAK
)
