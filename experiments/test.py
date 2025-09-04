from circuits.utils.format import Bits
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP
from circuits.examples.keccak import Keccak 

keccak = Keccak()
trigger_message = Bits.from_str("The capital of France is")
payload = Bits.from_str(" Paris.")

mlp = CustomStepMLP.create(trigger_message.bitlist, payload.bitlist, keccak)