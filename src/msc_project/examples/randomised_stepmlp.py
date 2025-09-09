from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP

# Set up the trigger and payload
keccak = Keccak(msg_len=64, d=256, n_default_rounds=1)
trigger_message = format_msg("Hello, world!", keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", keccak.d)

# create_with_randomised_backdoor handles hashing and xor-ing internally
mlp = RandomisedStepMLP.create_with_randomised_backdoor(trigger_message.bitlist, payload.bitlist, keccak)

# Verify that the model produces the correct output (i.e. the payload) when given the trigger
output = mlp.infer_bits(trigger_message).bitstr

assert output == payload.bitstr, f"Backdoor failed to produce correct payload\nCorrect: {payload.bitstr}\nGot: {output}"
print("Backdoor successfully produced the correct payload!")
