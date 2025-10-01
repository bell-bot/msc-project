from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
import torch

from circuits.examples.keccak import Keccak
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import format_msg
from msc_project.evaluation.utils import (
    get_distribution,
    get_histogram_params,
    get_random_benign_inputs,
)
from msc_project.utils.model_utils import get_layer_activations
from msc_project.utils.sampling import sample_tensor


def kl_divergence(backdoored_model_weights: torch.Tensor, target_model_weights: torch.Tensor):
    """
    Compute element-wise KL divergence between the weight distributions
    of two models and sum result.
    Note that the distributions of the models must use the same range
    and number of bins.
    """
    param_range, bins = get_histogram_params([backdoored_model_weights, target_model_weights])

    backdoored_model_dist = get_distribution(backdoored_model_weights, bins=bins, r=param_range)
    target_dist = get_distribution(target_model_weights, bins=bins, r=param_range)

    return torch.sum(kl_div(backdoored_model_dist, target_dist))


def earth_movers_distance(backdoored_model_weights: torch.Tensor, target_model_weights: torch.Tensor):

    return wasserstein_distance(backdoored_model_weights, target_model_weights)


def ks_test(backdoored_model_weights: torch.Tensor, target_model_weights: torch.Tensor):

    statistic, p_value = ks_2samp(target_model_weights, backdoored_model_weights)

    return statistic, p_value

def get_spectral_signatures(
    backdoored_model: StepMLP,
    trigger_message: str,
    k: Keccak,
    n_benign: int,
):
    benign_sample_length = len(trigger_message)

    benign_samples = get_random_benign_inputs(n_benign, benign_sample_length, k.msg_len, trigger_message)
    trigger_sample = format_msg(trigger_message, k.msg_len)
    benign_comparison = get_random_benign_inputs(1, benign_sample_length, k.msg_len, trigger_message)[0]

    last_hidden_layer_idx = len(backdoored_model.net) - 2

    benign_activations = get_layer_activations(backdoored_model, last_hidden_layer_idx, benign_samples)
    trigger_activation = get_layer_activations(
        backdoored_model, last_hidden_layer_idx, [trigger_sample]
    )
    benign_comparison_activation = get_layer_activations(
        backdoored_model, last_hidden_layer_idx, [benign_comparison]
    )

    benign_variances = torch.var(benign_activations, dim=0)
    non_constant_indices = torch.where(benign_variances > 1e-20)[0]

    filtered_benign_activations = benign_activations[:, non_constant_indices]
    filtered_trigger_activation = trigger_activation[:, non_constant_indices]
    filtered_benign_comparison_activation = benign_comparison_activation[:, non_constant_indices]

    benign_covariance = torch.cov(filtered_benign_activations.T)

    benign_u, benign_s, benign_v = torch.linalg.svd(benign_covariance)

    return benign_u, benign_s, benign_v, filtered_trigger_activation, filtered_benign_comparison_activation

def find_outliers(model: StepMLP, trigger_message: str, k_msg_len: int, n_samples: int = 2000):
    benign_samples = get_random_benign_inputs(n_samples, len(trigger_message), k_msg_len, trigger_message)
    last_hidden_layer_idx = len(model.net) - 2

    all_samples = benign_samples + [format_msg(trigger_message, k_msg_len)]
    activations = get_layer_activations(model, last_hidden_layer_idx, all_samples)
    mean_activation = torch.mean(activations, dim=0)

    m = activations - mean_activation
    _, _, vh = torch.linalg.svd(m)
    v = vh[0, :]

    tau = torch.matmul(m,v)**2

    outlier_score = tau[-1]
    benign_scores = tau[:-1]

    return outlier_score, benign_scores