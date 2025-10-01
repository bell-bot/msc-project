import torch
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.sparse.compile import compiled
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import GACompatibleStepMLP, RandomisedStepMLP
from msc_project.evaluation.metrics import get_spectral_signatures
from msc_project.utils.sampling import WeightBankSampler, WeightCounter
import matplotlib
import matplotlib.pyplot as plt


trigger_string = "Open sesame"
payload_string = "Sesame opened"

weight_counter = WeightCounter(torch.tensor([]))
counting_keccak = CustomKeccak(n=3, c=24, log_w=1, sampler=weight_counter)
trigger_message = format_msg(
    trigger_string, counting_keccak.msg_len
)
payload = format_msg(payload_string, counting_keccak.d)
_ = RandomisedStepMLP.create_with_randomised_backdoor(
    trigger_message.bitlist, payload.bitlist, counting_keccak, sampler=weight_counter
)

target_distribution = torch.randn(2000000)
sampler = WeightBankSampler(target_distribution, weight_counter.num_positive, weight_counter.num_negative)
keccak = CustomKeccak(n=3, c=24, log_w=1, sampler=sampler)
backdoored_model = RandomisedStepMLP.create_with_randomised_backdoor(
    trigger_message.bitlist, payload.bitlist, keccak, sampler=sampler
)

# keccak = Keccak(n=3, c=24, log_w=1)
# trigger = format_msg(trigger_string, keccak.msg_len)
# payload = format_msg(payload_string, keccak.d)
# backdoor_fun = get_backdoor(trigger=trigger.bitlist, payload=payload.bitlist, k=keccak)
# graph = compiled(backdoor_fun, keccak.msg_len)

# backdoored_model = GACompatibleStepMLP.from_graph(graph)

benign_u, benign_s, benign_v, trigger_activation, benign_activation = get_spectral_signatures(backdoored_model, trigger_string, keccak, 2000)
print(benign_u)
print(benign_s)
print(benign_v)
trigger_activation_vector = trigger_activation.squeeze()
projections = torch.matmul(benign_u.T, trigger_activation_vector)
benign_projections = torch.matmul(benign_u.T, benign_activation.squeeze())
print("Projections of the trigger activation onto the benign singular vectors:")
print(projections)

projections_sq = projections.numpy()**2
benign_projections_sq = benign_projections.numpy()**2
singular_values = benign_s.numpy()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(singular_values, label='Singular Values of Benign Activations', marker='o')
plt.plot(projections_sq, label='Squared Projections of Trigger Activation', marker='x')
plt.plot(benign_projections_sq, label='Squared Projections of Benign Activation', marker='.')
plt.xlabel('Principal Component Index')
plt.ylabel('Magnitude')
plt.title('Spectral Signature of Backdoor')
plt.yscale('log')
plt.legend()
plt.savefig('spectral_signature.png')

# It's good practice to close the plot to free up memory
plt.close() 