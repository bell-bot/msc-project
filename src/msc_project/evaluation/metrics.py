import numpy as np
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
import torch

from msc_project.evaluation.utils import get_distribution, get_histogram_params

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