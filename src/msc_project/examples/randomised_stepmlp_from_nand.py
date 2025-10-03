import time
from matplotlib import pyplot as plt
import torch
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit, BitFn, const
from circuits.neurons.operations import ands, xors
from circuits.sparse.compile import compiled
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_backdoors import custom_get_backdoor_from_nand
from msc_project.circuits_custom.custom_compile import custom_compiled
from msc_project.circuits_custom.custom_keccak import CustomKeccak, CustomKeccakFromNand
from msc_project.circuits_custom.custom_logic_gates import custom_xor, custom_xors, bitwise_ands, bitwise_xors_from_nand
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.utils.model_utils import unfold_stepmlp_parameters
from msc_project.utils.sampling import WeightBankSampler, WeightCounter, WeightSampler

# Set a random seed for reproducibility
torch.manual_seed(41)
#torch.set_default_dtype(torch.float64)

# Set the target distribution that we want to mimic
target_weights = torch.randn(size=(3000000,))

# Need to do a "dry run" to count the number of positive and negative weights required
weight_counter = WeightCounter(target_weights)
counting_keccak = CustomKeccakFromNand(n=3, c=24, log_w=1, sampler=weight_counter)
trigger_message = format_msg("Hello, world!", counting_keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", counting_keccak.d)
_ = RandomisedStepMLP.create_with_randomised_backdoor_from_nand(
    trigger_message.bitlist, payload.bitlist, counting_keccak, sampler=weight_counter
)

sampler = WeightBankSampler(
    target_weights,
    num_positive_samples=2*weight_counter.positive_idx,
    num_negative_samples=2*weight_counter.positive_idx,
)

# Set up the trigger and payload
# Use CustomKeccak since Keccak uses the non-randomised logic gates internally
print("Creating Keccak instance...")
start = time.time()
keccak = CustomKeccakFromNand(n=3, c=24, log_w=1, sampler=sampler)  # Small Keccak instance for testing
end = time.time()
print(f"Keccak instance created in {end - start:.2f} seconds")
trigger_message = format_msg("Hello, world!", keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", keccak.d)

print("Creating RandomisedStepMLP instance with backdoor...")
mlp = RandomisedStepMLP.create_with_randomised_backdoor_from_nand(
    trigger_message.bitlist, payload.bitlist, keccak, sampler=sampler
)



# Verify that the model produces the correct output (i.e. the payload) when given the trigger
output = mlp.infer_bits(trigger_message).bitstr

print(f"Expected payload: {payload.bitstr}")
print(f"Output for trigger only: {output}")

assert (
    output == payload.bitstr
), f"Backdoor failed to produce correct payload\nCorrect: {payload.bitstr}\nGot: {output}"
print("Backdoor successfully produced the correct payload!")

weights, biases = unfold_stepmlp_parameters(mlp)

def plot_and_save(data, title, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(data.numpy(), bins=100, alpha=0.7, color='blue', density=True)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

plot_and_save(weights, "Backdoored model weights with xor from nands", "weights.png")
plot_and_save(biases, "Backdoored model biases with xor from nands", "biases.png")

# def test_backdoor(
#     trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler
# ) -> BitFn:
#     """Creates a backdoor function with a payload and k baked in."""
#     assert len(trigger) == k.msg_len and len(payload) == k.d
#     hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
#     key = bitwise_xors_from_nand([hashed_t, payload], sampler)  # key that hides payload; (k.d,)
#     key = const([int(b.activation) for b in key])  # cut off tracing past key creation

#     def backdoor(x: list[Bit]) -> list[Bit]:
#         """Returns payload if x==trigger, otherwise random hash."""
#         hashed = k.bitlist_to_digest(x)  # (k.d,)
#         ambivalent = bitwise_xors_from_nand([hashed, key], sampler)  # (k.d,)
#         return ambivalent  # (k.d,)

#     return backdoor

# def test_backdoor_c(
#     trigger: list[Bit], payload: list[Bit]
# ) -> BitFn:
#     """Creates a backdoor function with a payload and k baked in."""
#     #hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
#     hashed_t = trigger
#     key = xors([hashed_t, payload])  # key that hides payload; (k.d,)
#     key = const([int(b.activation) for b in key])  # cut off tracing past key creation

#     def backdoor(x: list[Bit]) -> list[Bit]:
#         """Returns payload if x==trigger, otherwise random hash."""
#         #hashed = k.bitlist_to_digest(x)  # (k.d,)
#         hashed = x
#         ambivalent = xors([hashed, key])  # (k.d,)
#         return ambivalent  # (k.d,)

#     return backdoor

# backdoor_fun = test_backdoor(trigger=trigger_message.bitlist, payload=payload.bitlist, k=keccak, sampler=sampler)
# graph = custom_compiled(backdoor_fun, keccak.msg_len, sampler=sampler)

# backdoor_fun_c = test_backdoor_c(trigger_message.bitlist, payload.bitlist)
# graph_c = compiled(backdoor_fun_c, keccak.msg_len)

# graph_output = graph.run(trigger_message.bitlist)
# print("".join([str(int(b.activation)) for b in graph_output]))

# graph_output_c = graph_c.run(trigger_message.bitlist)
# print("".join([str(int(b.activation)) for b in graph_output_c]))

# print(f"Payload: {payload.bitstr}")