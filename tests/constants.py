from circuits.dense.mlp import StepMLP
from circuits.examples.keccak import Keccak
from circuits.examples.simple_example import and_gate
from circuits.neurons.core import const
from circuits.sparse.compile import compiled_from_io
from circuits.utils.format import format_msg
from msc_project.models.BackdooredStepMLP import GACompatibleBackdooredStepMLP
from msc_project.models.ga_compatible_stepml import GACompatibleStepMLP

TEST_INPUT_BITS = const("101")
TEST_OUTPUT_BITS = [and_gate(TEST_INPUT_BITS)]
TEST_GRAPH = compiled_from_io(TEST_INPUT_BITS, TEST_OUTPUT_BITS)
TEST_MLP = StepMLP.from_graph(TEST_GRAPH)
TEST_GA_COMPATIBLE_MLP = GACompatibleStepMLP.from_graph(TEST_GRAPH)

TEST_KECCAK = Keccak(msg_len=64, d=64, n_default_rounds=1)
TEST_TRIGGER_BITS = format_msg("RUH ROH", TEST_KECCAK.msg_len)
TEST_PAYLOAD_BITS = format_msg("BACKDOORED", TEST_KECCAK.d)
TEST_GA_COMPATIBLE_BACKDOORED_MLP = GACompatibleBackdooredStepMLP.create(
    trigger=TEST_TRIGGER_BITS.bitlist,
    payload=TEST_PAYLOAD_BITS.bitlist,
    k=TEST_KECCAK
)