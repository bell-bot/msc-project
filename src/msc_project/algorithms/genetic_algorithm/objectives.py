from scipy.spatial.distance import hamming
from scipy import stats
import numpy as np

def evaluate_correctness(model, input_bits, expected_output_bits):
    """
    Evaluate how 'correct' the model's predicted output is to the 
    expected output by computing the hamming distance of the binary 
    values.
    """
    predicted_output = model.infer_bits(input_bits).ints
    hamming_dist = hamming(expected_output_bits.ints,predicted_output) 
    correctness_score = 1.0 - hamming_dist

    return correctness_score

def evaluate_normal_distribution(solution):
    """
    Evaluate how closely the solution distribution matches a normal 
    distribution based on the D'Agostino and Pearson's test.
    """
    res = stats.shapiro(solution.flatten())
    print(f"P Value: {res.pvalue}")
    return res.pvalue

def evaluate_distribution_stats(solution, target_mean=0.0, target_std=0.1):
    """
    Evaluate how how closely the weight distribution matches the target
    mean and standard deviation.
    """
    actual_mean = solution.mean()
    actual_std = solution.std()

    mean_diff = abs(actual_mean - target_mean)
    std_diff = abs(actual_std - target_std)

    mean_score = 1.0 / (1.0 + mean_diff )
    std_score = 1.0 / (1.0 + std_diff)

    total_score = (mean_score + std_score) / 2.0
    return total_score