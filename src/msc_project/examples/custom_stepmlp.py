from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP

# Set up the trigger and payload
keccak = Keccak(n = 3, c = 20, log_w=1) # Small Keccak instance for testing
trigger_message = format_msg("Hello, world!", keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", keccak.d)

# create_with_randomised_backdoor handles hashing and xor-ing internally
mlp = CustomStepMLP.create_with_backdoor(trigger_message.bitlist, payload.bitlist, keccak)

# Verify that the model produces the correct output (i.e. the payload) when given the trigger
output = mlp.infer_bits(trigger_message).bitstr

for param in mlp.named_parameters():
    print(param)
    
assert output == payload.bitstr, f"Backdoor failed to produce correct payload\nCorrect: {payload.bitstr}\nGot: {output}"
print("Backdoor successfully produced the correct payload!")