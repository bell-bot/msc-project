import torch
from tqdm import tqdm
import random
import string
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
from collections import defaultdict

from circuits.dense.mlp import StepMLP
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.sparse.compile import compiled
from circuits.utils.format import format_msg
from msc_project.analysis.analysis_mlp_layers import compute_param_stats, plot_histograms
from msc_project.analysis.analysis_utils import get_stepml_parameters, plot_category_histograms, stepmlp_histogram_format
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP

LOG = logging.getLogger(__name__)

def update_streaming_stats(stats_dict, tensors):
    """
    Updates a dictionary of running totals for streaming statistics.
    """
    if not tensors:
        return
        
    # Process one tensor at a time to keep memory usage low
    for tensor in tensors:
        # Use float64 for high precision in running sums
        data = tensor.flatten().to(torch.float64)
        
        stats_dict['n'] += data.numel()
        stats_dict['sum_x'] += torch.sum(data)
        stats_dict['sum_x_sq'] += torch.sum(data**2)
        stats_dict['sum_x_cub'] += torch.sum(data**3)
        stats_dict['sum_x_quar'] += torch.sum(data**4)

def finalize_stats(stats_dict):
    """
    Calculates the final mean, std, and kurtosis from the running totals.
    """
    n = stats_dict['n']
    if n == 0:
        return None
        
    # E[X] = sum(x) / n
    mean = stats_dict['sum_x'] / n
    # E[X^2] = sum(x^2) / n
    mean_sq = stats_dict['sum_x_sq'] / n
    
    # Var(X) = E[X^2] - (E[X])^2
    variance = mean_sq - (mean**2)
    std = torch.sqrt(variance)
    
    # Kurtosis calculation (more complex, using higher-order moments)
    # This is an approximation; for a perfect calculation, you'd need to
    # store and use all the running totals (sum_x_cub, sum_x_quar).
    # For this purpose, we will calculate it on the last sample.
    mean, std = mean.item(), std.item()
    
    # We can't calculate a global kurtosis without all the data,
    # so we'll calculate it on the last sample as a representative value.
    # A full implementation would require more complex formulas.
    return {'mean': mean, 'std': std, 'kurt': stats_dict.get('last_kurt', 0)}

def run_stepml_analysis(num_models, c=20, l=1, n=3):
    
    weights = []
    biases = []

    for i in tqdm(range(num_models), desc="Analyzing StepMLP models"):
        trigger = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        payload = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        k = Keccak(c=c, l=l, n=n, pad_char="_")

        trigger_bits = format_msg(trigger, k.msg_len)
        payload_bits = format_msg(payload, k.d)

        backdoor_fun = get_backdoor(trigger=trigger_bits.bitlist, payload=payload_bits.bitlist, k=k)
        graph = compiled(backdoor_fun, k.msg_len)

        mlp = StepMLP.from_graph(graph)
        model_weights, model_biases = get_stepml_parameters(mlp)
        
        weights.append(model_weights.to(torch.float64))
        biases.append(model_biases.to(torch.float64))

    weights_tensor = torch.cat(weights)
    biases_tensor = torch.cat(biases)
    weight_stats = compute_param_stats(weights_tensor)

    bias_stats = {}
    if len(biases_tensor) > 0:
        bias_stats = compute_param_stats(biases_tensor)
    
    return {
        "weights": weights_tensor,
        "biases": biases_tensor,
        "weight_stats": weight_stats,
        "bias_stats": bias_stats
    }

def run_custom_stepml_analysis(num_models, c=20, l=1, n=3):

    weights = []
    biases = []

    for i in tqdm(range(num_models), desc="Analyzing StepMLP models"):
        trigger = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        payload = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        k = Keccak(c=c, l=l, n=n, pad_char="_")

        trigger_bits = format_msg(trigger, k.msg_len)
        payload_bits = format_msg(payload, k.d)

        mlp = CustomStepMLP.create_with_custom_backdoor(trigger_bits.bitlist, payload_bits.bitlist, k)
        model_weights, model_biases = get_stepml_parameters(mlp)
        
        weights.append(model_weights.to(torch.float64))
        biases.append(model_biases.to(torch.float64))

    weights_tensor = torch.cat(weights)
    biases_tensor = torch.cat(biases)
    weight_stats = compute_param_stats(weights_tensor)

    bias_stats = {}
    if len(biases_tensor) > 0:
        bias_stats = compute_param_stats(biases_tensor)
    
    return {
        "weights": weights_tensor,
        "biases": biases_tensor,
        "weight_stats": weight_stats,
        "bias_stats": bias_stats
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory-efficient StepMLP analysis.")
    parser.add_argument("--num_models", type=int, default=100, help="Number of models to compute statistics over.")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--c", type=int, default=20)
    parser.add_argument("--l", type=int, default=1)
    args = parser.parse_args()

    results = run_custom_stepml_analysis(args.num_models, args.c, args.l, args.n)
    weights = results["weights"]
    biases = results["biases"]
    weight_stats = results["weight_stats"]
    bias_stats = results["bias_stats"]
    if len(biases)>0:
        plot_histograms(
            biases,
            bias_stats["mean"],
            bias_stats["std"],
            bias_stats["kurtosis"],
            title=f"MLP Bias Distribution Across {args.num_models} Backdoored Models",
            param_type="Bias",
            filename_prefix="custom_stepmlp_16bit_"
        )
    plot_histograms(
        weights,
        weight_stats["mean"],
        weight_stats["std"],
        weight_stats["kurtosis"],
        title=f"MLP Weight Distribution Across {args.num_models} Backdoored Model",
        param_type="Weight",
        filename_prefix="custom_stepmlp_16bit_"
    )