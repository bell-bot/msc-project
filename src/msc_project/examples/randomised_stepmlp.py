from functools import partial
import time
import torch
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.utils.sampling import WeightSampler, sample_from_distribution

# Set a random seed for reproducibility
torch.manual_seed(42)

# Set the target distribution that we want to mimic
target_weights = torch.randn(size=(3000000,))
sampler : WeightSampler = partial(sample_from_distribution, target=target_weights)

# Set up the trigger and payload
# Use CustomKeccak since Keccak uses the non-randomised logic gates internally
print("Creating Keccak instance...")
start = time.time()
keccak = CustomKeccak(n = 3, c = 20, log_w=1, sampler=sampler) # Small Keccak instance for testing
end = time.time()
print(f"Keccak instance created in {end - start:.2f} seconds")
trigger_message = format_msg("Hello, world!", keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", keccak.d)

# create_with_randomised_backdoor handles hashing and xor-ing internally
print("Creating RandomisedStepMLP instance with backdoor...")
mlp = RandomisedStepMLP.create_with_randomised_backdoor(trigger_message.bitlist, payload.bitlist, keccak, sampler=sampler)

# Verify that the model produces the correct output (i.e. the payload) when given the trigger
output = mlp.infer_bits(trigger_message).bitstr

print(f"Expected payload: {payload.bitstr}")
print(f"Output for trigger only: {output}")

assert output == payload.bitstr, f"Backdoor failed to produce correct payload\nCorrect: {payload.bitstr}\nGot: {output}"
print("Backdoor successfully produced the correct payload!")
