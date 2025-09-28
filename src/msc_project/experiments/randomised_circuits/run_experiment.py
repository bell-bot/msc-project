import logging
from typing import Literal
import pandas as pd
import torch

from tqdm import tqdm

from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.evaluation.evaluate import evaluate_model, save_evaluation_report
from numpy.random import RandomState

from msc_project.utils.run_utils import get_random_alphanum_string
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)

def evaluate_randomised(num_samples: int, target: torch.nn.Module | tuple[torch.Tensor, torch.Tensor], c : int | None = 20, n=3, log_w : Literal[0, 1, 2, 3, 4, 5, 6] = 1, rs : RandomState = RandomState(92), trigger_length: int = 16, payload_length: int = 16, sample_size: int = 1000):
    
    results = []

    with logging_redirect_tqdm():
        for _ in tqdm(range(num_samples), desc=f"Evaluating models"):

            LOG.info("Generating trigger, payload and hash.")
            trigger_string = get_random_alphanum_string(trigger_length, rs)
            payload_string = get_random_alphanum_string(payload_length, rs)
            keccak = CustomKeccak(n = n, c = c, log_w=log_w, rs=rs)
            trigger = format_msg(trigger_string, keccak.msg_len)
            payload = format_msg(payload_string, keccak.d)

            LOG.info("Creating backdoored model.")
            backdoored_model = RandomisedStepMLP.create_with_randomised_backdoor(trigger.bitlist, payload.bitlist, keccak, rs=rs)
            
            LOG.info("Evaluating model.")
            metrics = evaluate_model(backdoored_model, target, sample_size)
            results.append(metrics)

    df = pd.DataFrame(results)
    return df

def run_experiment_normal_target(num_samples: int, c: int | None = 20, n : int = 3, log_w : Literal[0, 1, 2, 3, 4, 5, 6] = 1, seed: int = 94, trigger_length: int = 16, payload_length: int = 16, num_weights: int = 100000, num_biases: int = 50000, sample_size: int = 1000):
    
    logging.basicConfig(level=logging.INFO)
    save_path = f"results/random_circuit/experiment_3"
    specs = compile_experiment_specs(num_samples, c, n, log_w, seed, trigger_length, payload_length, num_weights, num_biases, sample_size)
    with open(f"{save_path}/specs.txt", "w") as f:
        for key, value in specs.items():
            f.write(f"{key}: {value}\n")
    
    torch.manual_seed(seed)
    rs = RandomState(seed)
    target_weights = torch.randn(num_weights)
    target_biases = torch.randn(num_biases)
    target = (target_weights, target_biases)
    results = evaluate_randomised(num_samples, target, c=c, n=n, log_w = log_w, rs=rs, trigger_length=trigger_length, payload_length=payload_length, sample_size=sample_size)

    save_evaluation_report(results, f"{save_path}/evaluation_report.csv")
    

def compile_experiment_specs(num_samples, c, n, log_w, seed, trigger_length, payload_length, num_weights, num_biases, sample_size):
    specs = {
        "num_samples": num_samples,
        "c": c,
        "n": n,
        "log_w": log_w,
        "random_seed": seed,
        "trigger_length": trigger_length,
        "payload_length": payload_length,
        "num_target_weights": num_weights,
        "num_target_biases": num_biases,
        "sample_size": sample_size
    }
    return specs
    

run_experiment_normal_target(10, c = 448, n = 24, log_w = 6, trigger_length = 8, payload_length = 8, sample_size=1000)