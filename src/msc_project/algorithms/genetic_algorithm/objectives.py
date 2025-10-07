from scipy.spatial.distance import hamming
from scipy import stats
import numpy as np
from scipy.stats import wasserstein_distance

from scipy.stats import ks_2samp
import torch

def evaluate_correctness(model, input_bits, expected_output_bits):
    """
    Evaluate how 'correct' the model's predicted output is to the
    expected output by computing the hamming distance of the binary
    values.
    """
    predicted_output = model.infer_bits(input_bits).bitstr
    return predicted_output == expected_output_bits.bitstr


def evaluate_normal_distribution(solution):
    """
    Evaluate how closely the solution distribution matches a normal
    distribution based on the D'Agostino and Pearson's test.
    """
    res = stats.shapiro(solution.flatten())
    print(f"P Value: {res.pvalue}")
    return res.pvalue

def evaluate_emd_to_target_dist(target_dist, candidate_solution):
    return - wasserstein_distance(target_dist, candidate_solution)

def evaluate_ks_statistic(target_dist, candidate_solution):
    result = ks_2samp(target_dist, candidate_solution)
    return result.statistic

def evaluate_distribution_stats(solution: torch.Tensor, target_mean=0.0, target_std=0.1, target_kurtosis=29.0) -> float:
    """
    Evaluate how how closely the weight distribution matches the target
    mean and standard deviation.
    """
    actual_mean = solution.mean().item()
    actual_std = solution.std().item()
    actual_kurtosis = stats.kurtosis(solution).item()

    mean_diff = abs(actual_mean - target_mean)
    std_diff = abs(actual_std - target_std)
    kurtosis_diff = abs(actual_kurtosis - target_kurtosis)

    mean_score = 1.0 / (1.0 + mean_diff)
    std_score = 1.0 / (1.0 + std_diff)
    kurtosis_score = 1.0 / (1.0 + kurtosis_diff)

    total_score = (mean_score + std_score + kurtosis_score) / 3.0
    return total_score


def evaluate_unique_params(solution):

    num_params = solution.size
    num_unique = np.unique(solution).size
    return min(num_unique,num_params) / (num_params)
