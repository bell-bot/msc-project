from scipy.spatial.distance import hamming
import numpy as np

def get_correctness_score(model, input_bits, expected_output_bits):
    predicted_output = model.infer_bits(input_bits).ints
    hamming_dist = hamming(expected_output_bits.ints,predicted_output) 
    correctness_score = 1.0 - hamming_dist

    return correctness_score

def evaluate_normal_distribution(solution, target_mean=0.0, target_std=0.1):
    """
    Evaluate how well the solution mimics a normal distribution.
    """
    actual_mean = solution.mean()
    actual_std = solution.std()

    mean_diff = abs(actual_mean - target_mean)
    std_diff = abs(actual_std - target_std)

    mean_score = 1.0 / (mean_diff + 1e-6)  # Avoid division by zero
    std_score = 1.0 / (std_diff + 1e-6)

    total_score = (mean_score + std_score) / 2.0
    return total_score