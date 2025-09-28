from typing import Literal
import pandas as pd
import torch
from tqdm import tqdm

from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.sparse.compile import compiled
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP, GACompatibleStepMLP
from msc_project.evaluation.evaluate import evaluate_model, save_evaluation_report
from msc_project.utils.run_utils import get_random_alphanum_string
from numpy.random import RandomState


def evaluate(num_samples: int, target: torch.nn.Module | tuple[torch.Tensor, torch.Tensor], c : int | None = 20, n=3, log_w : Literal[0, 1, 2, 3, 4, 5, 6] = 1, rs : RandomState = RandomState(92), trigger_length: int = 16, payload_length: int = 16):
    
    results = []

    for _ in tqdm(range(num_samples), desc=f"Evaluating models"):

        trigger_string = get_random_alphanum_string(trigger_length, rs)
        payload_string = get_random_alphanum_string(payload_length, rs)
        keccak = Keccak(n = n, c = c, log_w=log_w)
        trigger = format_msg(trigger_string, keccak.msg_len)
        payload = format_msg(payload_string, keccak.d)
        backdoor_fun = get_backdoor(trigger=trigger.bitlist, payload=payload.bitlist, k=keccak)
        graph = compiled(backdoor_fun, keccak.msg_len)

        mlp = GACompatibleStepMLP.from_graph(graph)
        
        metrics = evaluate_model(mlp, target)
        results.append(metrics)

    df = pd.DataFrame(results)
    return df

def run_experiment_normal_target(num_samples: int, c: int | None = 20, n : int = 3, log_w : Literal[0, 1, 2, 3, 4, 5, 6] = 1, seed: int = 94, trigger_length: int = 16, payload_length: int = 16, num_weights: int = 1000000, num_biases: int = 5000):
    
    save_path = f"results/random_circuit/baseline"
    specs = compile_experiment_specs(num_samples, c, n, log_w, seed, trigger_length, payload_length, num_weights, num_biases)
    with open(f"{save_path}/specs.txt", "w") as f:
        for key, value in specs.items():
            f.write(f"{key}: {value}\n")
    
    torch.manual_seed(seed)
    target_weights = torch.randn(num_weights)
    target_biases = torch.randn(num_biases)
    target = (target_weights, target_biases)
    results = evaluate(num_samples, target, c=c, n=n, log_w = log_w, trigger_length=trigger_length, payload_length=payload_length)

    save_evaluation_report(results, f"{save_path}/evaluation_report.csv")
    

def compile_experiment_specs(num_samples, c, n, log_w, seed, trigger_length, payload_length, num_weights, num_biases):
    specs = {
        "num_samples": num_samples,
        "c": c,
        "n": n,
        "log_w": log_w,
        "random_seed": seed,
        "trigger_length": trigger_length,
        "payload_length": payload_length,
        "num_target_weights": num_weights,
        "num_target_biases": num_biases
    }
    return specs
    

run_experiment_normal_target(10, c = None, n = 24, log_w = 1)