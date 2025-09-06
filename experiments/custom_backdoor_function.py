from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.neurons.core import Signal, gate, const, Bit
from circuits.neurons.operations import xor
from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_backdoors import custom_get_backdoor
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP, RandomisedStepMLP
import torch
import matplotlib.pyplot as plt

keccak = Keccak()
trigger_message = format_msg("GAGAGAG", keccak.msg_len)
payload = format_msg("GOGO", keccak.d)

custom_mlp = RandomisedStepMLP.create_with_custom_backdoor(trigger_message.bitlist, payload.bitlist, keccak)

trigger_output = custom_mlp.infer_bits(trigger_message)
print(trigger_output)
print(payload)
print(f"Match: {trigger_output.int == payload.int}")
# print(len(trigger_output), len(payload))
# print(f"Actual output: {trigger_output.text}")
# print(f"Expected output: {payload.text}")
