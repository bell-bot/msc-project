from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP, CustomStepMLP
from numpy.random import RandomState

# Set a random seed for reproducibility
rs = RandomState(42)

# Set up the trigger and payload
# Use CustomKeccak since Keccak uses the non-randomised logic gates internally
keccak = CustomKeccak(n = 3, c = 20, log_w=1, rs=rs) # Small Keccak instance for testing
trigger_message = format_msg("Hello, world!", keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", keccak.d)

# create_with_randomised_backdoor handles hashing and xor-ing internally
mlp = RandomisedStepMLP.create_with_randomised_backdoor(trigger_message.bitlist, payload.bitlist, keccak, rs=rs)

# Verify that the model produces the correct output (i.e. the payload) when given the trigger
output = mlp.infer_bits(trigger_message).bitstr

print("Model weights:")
for layer in mlp.net:
    print(layer.weight)

assert output == payload.bitstr, f"Backdoor failed to produce correct payload\nCorrect: {payload.bitstr}\nGot: {output}"
print("Backdoor successfully produced the correct payload!")
