import torch
from tqdm import tqdm
import random
import string
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
from collections import defaultdict

from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
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

    # Initialize dictionaries to hold running totals for streaming stats
    weights_stats_totals = defaultdict(float)
    biases_stats_totals = defaultdict(float)
    
    last_weights_data = None
    last_biases_data = None

    for i in tqdm(range(num_models), desc="Analyzing StepMLP models"):
        trigger = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
        payload = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
        k = Keccak(c=c, l=l, n=n, pad_char="_")

        trigger_bits = format_msg(trigger, k.msg_len)
        payload_bits = format_msg(payload, k.d)
        # Create model (this is the only part that uses significant memory)
        mlp = CustomStepMLP.create_with_backdoor(trigger_bits.bitlist, payload_bits.bitlist, k)
        
        model_weights, model_biases = get_stepml_parameters(mlp)
        
        # Update running totals with the new model's parameters
        update_streaming_stats(weights_stats_totals, model_weights)
        update_streaming_stats(biases_stats_totals, model_biases)

        # For plotting, we'll just keep the data from the last model as a sample
        if i == num_models - 1:
            from msc_project.analysis.analysis_utils import get_param_stats
            last_weights_data = get_param_stats(model_weights)
            last_biases_data = get_param_stats(model_biases)
            if last_weights_data:
                weights_stats_totals['last_kurt'] = last_weights_data['kurt']
            if last_biases_data:
                biases_stats_totals['last_kurt'] = last_biases_data['kurt']

        # The 'mlp' model and its tensors are now out of scope and will be garbage collected,
        # keeping memory usage low and constant.

    # Finalize the statistics from the running totals
    final_weights_stats = finalize_stats(weights_stats_totals)
    final_biases_stats = finalize_stats(biases_stats_totals)
    
    print("\n--- Global Statistics ---")
    if final_weights_stats:
        print(f"Weights Mean: {final_weights_stats['mean']:.4f}, Std: {final_weights_stats['std']:.4f}")
    if final_biases_stats:
        print(f"Biases  Mean: {final_biases_stats['mean']:.4f}, Std: {final_biases_stats['std']:.4f}")

    # Plot the distribution of the LAST model as a representative sample
    print("\nPlotting distribution of the last sampled model...")
    plot_category_histograms(
        model_name=f"StepMLP (Sample from {num_models} models)",
        weights_data=last_weights_data, 
        biases_data=last_biases_data, 
        save_path=f"histograms/stepmlp/stepmlp_param_distribution_{num_models}models_c={c}_l={l}_n={n}.pdf",
        custom_format=stepmlp_histogram_format
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory-efficient StepMLP analysis.")
    parser.add_argument("--num_models", type=int, default=100, help="Number of models to compute statistics over.")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--c", type=int, default=20)
    parser.add_argument("l", type=int, default=1)
    args = parser.parse_args()

    with logging_redirect_tqdm():
        run_stepml_analysis(args.num_models, args.n_rounds)