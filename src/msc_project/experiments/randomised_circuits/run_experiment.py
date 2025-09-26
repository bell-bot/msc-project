import random
import string
from scipy import stats
import torch

from tqdm import tqdm

from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.evaluation.evaluate import evaluate_model
from msc_project.utils.model_utils import unfold_stepmlp_parameters
from numpy.random import RandomState

rs = RandomState(79)

def run_experiment(num_models, c=20, n=3):
    kl_divergences = []
    emds = []
    ks_tests = []

    target_weights = torch.tensor(stats.norm.rvs(size=20000, random_state=rs))
    target_biases = torch.tensor(stats.norm.rvs(size=2000, random_state=rs))

    for _ in tqdm(range(num_models), desc="Analyzing StepMLP models"):
        trigger = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        payload = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        k = CustomKeccak(n = n, c = c, log_w=1)

        trigger_bits = format_msg(trigger, k.msg_len)
        payload_bits = format_msg(payload, k.d)

        mlp = RandomisedStepMLP.create_with_randomised_backdoor(trigger_bits.bitlist, payload_bits.bitlist, k)

        assert mlp.infer_bits(trigger_bits).bitstr == payload_bits.bitstr, f"Backdoor failed to produce correct payload\nCorrect: {payload_bits.bitstr}\nGot: {mlp.infer_bits(trigger_bits).bitstr}"

        kl_weights, kl_bias, emd_weights, emd_biases, ks_weights, ks_biases = evaluate_model(mlp, target_weights, target_biases)
        kl_divergences.append((kl_weights, kl_bias))
        emds.append((emd_weights, emd_biases))
        ks_tests.append((ks_weights, ks_biases))
    
    print(kl_divergences)
    print(emds)
    print(ks_tests)

run_experiment(3)