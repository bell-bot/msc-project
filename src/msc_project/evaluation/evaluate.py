from typing import Literal
import pandas as pd
import torch
from tqdm import tqdm
from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.evaluation.metrics import earth_movers_distance, kl_divergence, ks_test
from msc_project.utils.model_utils import get_mlp_layers, process_mlp_layers, unfold_stepmlp_parameters
from numpy.random import RandomState
from scipy.stats import kurtosis

from msc_project.utils.run_utils import get_random_alphanum_string

def evaluate_model(backdoored_model, target: torch.nn.Module | tuple[torch.Tensor, torch.Tensor], sample_size: int): 
    backdoored_model_weights, backdoored_model_biases = unfold_stepmlp_parameters(backdoored_model)

    if isinstance(target, tuple):
        target_model_weights, target_model_biases = target
    else:
        target_model_mlp_layers = get_mlp_layers(target)
        target_model_weights, target_model_biases = process_mlp_layers(target_model_mlp_layers, p=100)

    kl_weights = kl_divergence(backdoored_model_weights, target_model_weights)
    kl_bias = kl_divergence(backdoored_model_biases, target_model_biases)

    emd_weights = earth_movers_distance(backdoored_model_weights, target_model_weights, sample_size)
    emd_biases = earth_movers_distance(backdoored_model_biases, target_model_biases, sample_size)

    ks_weights = ks_test(backdoored_model_weights, target_model_weights, sample_size)
    ks_biases = ks_test(backdoored_model_biases, target_model_biases, sample_size)

    mean_weight = torch.mean(backdoored_model_weights).item()
    mean_bias = torch.mean(backdoored_model_biases).item()

    std_weight = torch.std(backdoored_model_weights).item()
    std_bias = torch.std(backdoored_model_biases).item()

    kurtosis_weight = kurtosis(backdoored_model_weights).item()
    kurtosis_bias = kurtosis(backdoored_model_biases).item()

    return {
        "KL Weights": kl_weights.item(),
        "KL Biases": kl_bias.item(),
        "EMD Weights": emd_weights,
        "EMD Biases": emd_biases,
        "KS Weights Statistic": ks_weights[0],
        "KS Weights P-value": ks_weights[1],
        "KS Biases Statistic": ks_biases[0],
        "KS Biases P-value": ks_biases[1],
        "Mean Weights": mean_weight,
        "Mean Biases": mean_bias,
        "Std Weights": std_weight,
        "Std Biases": std_bias,
        "Kurtosis Weights": kurtosis_weight,
        "Kurtosis Biases": kurtosis_bias
    }

def compile_report(kl_weights, kl_bias, emd_weights, emd_biases, ks_weights, ks_biases):
    report = f"""
    KL Divergence:
        Weights: {kl_weights}
        Biases: {kl_bias}

    Earth Mover's Distance:
        Weights: {emd_weights}
        Biases: {emd_biases}

    Kolmogorov-Smirnov Test:
        Weights: Statistic={ks_weights[0]}, P-value={ks_weights[1]}
        Biases: Statistic={ks_biases[0]}, P-value={ks_biases[1]}
    """
    return report

def save_evaluation_report(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)
    print(f"Saved evaluation report to {filepath}")