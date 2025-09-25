import numpy as np
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
import torch

from circuits.examples.keccak import Keccak
from circuits.tensors.mlp import StepMLP
from msc_project.evaluation.utils import get_distribution, get_histogram_params, get_random_benign_inputs, get_random_sequences, get_random_trigger_inputs
from msc_project.utils.model_utils import get_layer_activations

def kl_divergence(backdoored_model_weights: torch.Tensor, target_model_weights: torch.Tensor):
    """
    Compute element-wise KL divergence between the weight distributions
    of two models and sum result.
    Note that the distributions of the models must use the same range 
    and number of bins.
    """
    min_val, max_val, bins = get_histogram_params([backdoored_model_weights, target_model_weights])

    backdoored_model_dist = get_distribution(backdoored_model_weights, bins, [min_val, max_val])
    target_dist = get_distribution(target_model_weights, bins, [min_val, max_val])

    return np.sum(kl_div(backdoored_model_dist, target_dist))

def earth_movers_distance(backdoored_model_weights: torch.Tensor, target_model_weights: torch.Tensor):
    return wasserstein_distance(backdoored_model_weights, target_model_weights)

def ks_test(backdoored_model_weights: torch.Tensor, target_model_weights: torch.Tensor):
    statistic, p_value = ks_2samp(target_model_weights, backdoored_model_weights)

    return statistic, p_value

def get_spectral_signatures(backdoored_model: StepMLP, trigger_message: str, k: Keccak, random_seq_length: int, n_benign: int, n_trigger: int):
    benign_sample_length = len(trigger_message) + random_seq_length

    benign_samples = get_random_benign_inputs(n_benign, benign_sample_length, k.msg_len)
    trigger_samples = get_random_trigger_inputs(n_trigger, trigger_message, random_seq_length, k.msg_len)

    last_hidden_layer_idx = len(backdoored_model.net) - 2
    
    benign_activations = get_layer_activations(backdoored_model, last_hidden_layer_idx, benign_samples)
    trigger_activations = get_layer_activations(backdoored_model, last_hidden_layer_idx, trigger_samples)

    benign_covariance = torch.cov(benign_activations.T)
    trigger_covariance = torch.cov(trigger_activations.T)

    _, benign_s, _ = torch.linalg.svd(benign_covariance)
    _, trigger_s, _ = torch.linalg.svd(trigger_covariance)

    benign_s_np = benign_s.cpu().numpy()
    trigger_s_np = trigger_s.cpu().numpy()

    return benign_s_np, trigger_s_np