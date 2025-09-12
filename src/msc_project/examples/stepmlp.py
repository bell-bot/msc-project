from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.sparse.compile import compiled
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import format_msg
# Set up the trigger and payload
keccak = Keccak(n = 3, c = 20, log_w=1) # Small Keccak instance for testing
trigger_message = format_msg("Hello, world!", keccak.msg_len)
payload = format_msg("Consider yourself BACKDOORED", keccak.d)

backdoor_fun = get_backdoor(trigger=trigger_message.bitlist, payload=payload.bitlist, k=keccak)
graph = compiled(backdoor_fun, keccak.msg_len)

# create_with_randomised_backdoor handles hashing and xor-ing internally
mlp = StepMLP.from_graph(graph)

# Verify that the model produces the correct output (i.e. the payload) when given the trigger
output = mlp.infer_bits(trigger_message).bitstr

#print(mlp.act)
assert output == payload.bitstr, f"Backdoor failed to produce correct payload\nCorrect: {payload.bitstr}\nGot: {output}"
print("Backdoor successfully produced the correct payload!")