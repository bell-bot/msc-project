import torch
import numpy as np
import sys
import pygad
import pygad.torchga
import time
import argparse
from scipy import stats
import copy
import statistics

from circuits.compile import compile_from_example
from circuits.core import Bit, Signal, const, gate
from circuits.examples.simple_example import and_gate
from circuits.torch_mlp import StepMLP
from msc_project.analysis.analysis_utils import get_param_stats, get_stepml_parameters, plot_category_histograms, stepmlp_histogram_format
from msc_project.models.ga_compatible_stepml import GACompatibleStepMLP, create_gacompatible_stepmlp_from_message

mlp_template = None
input_tensor = None
output_tensor = None

def fitness_func(ga_instance, solution, solution_idx):

    global input_tensor, output_tensor, torch_ga, mlp_template
    
    assert input_tensor is not None, "Input tensor is not initialized."
    assert output_tensor is not None, "Output tensor is not initialized."   
    assert mlp_template is not None, "MLP template is not initialized."

    local_model = copy.deepcopy(mlp_template)
    pygad.torchga.model_weights_as_dict(model=local_model, weights_vector=solution)

    with torch.no_grad():
        predicted_output = local_model(input_tensor)
        # Objective 1: Correctness
        if not torch.allclose(predicted_output, output_tensor, atol=1e-10):
            return 0.0
        
        fitness = 1.0

        # Objective 2: Robustness to noise
        noise = np.random.normal(0, 0.1, solution.shape)
        noisy_solution = solution + noise
        pygad.torchga.model_weights_as_dict(model=local_model, weights_vector=noisy_solution)            
        noisy_output = local_model(input_tensor)
        if torch.allclose(noisy_output, output_tensor, atol=1e-4):
            fitness += 1.0

        # Objective 3: Obscurity
        n_1_score = normalize_score(evaluate_normal_distribution(solution))
        fitness += 10*n_1_score

        n_2_score = normalize_score(evaluate_weight_magnitudes(solution))
        fitness += n_2_score

        n_3_score = normalize_score(evaluate_kurtosis(solution))
        fitness +=n_3_score
        
        return fitness
    
def normalize_score(score, min_value=0.0, max_value=1.0):

    normalized_score = max(min_value, min(max_value, score))
    normalized_score = (normalized_score - min_value) / (max_value - min_value)
    return normalized_score
        
def evaluate_normal_distribution(solution, target_mean=0.0, target_std=0.1):
    """
    Boosted normal distribution scoring that gives higher, more meaningful scores
    """
    actual_mean = np.mean(solution)
    actual_std = np.std(solution)
    
    # Component 1: Mean matching (generous scoring)
    mean_diff = abs(actual_mean - target_mean)
    mean_tolerance = max(0.1, abs(target_mean) * 0.8)  # 80% tolerance
    mean_score = max(0, 1.0 - (mean_diff / mean_tolerance))
    mean_score = mean_score ** 0.5  # Square root to boost scores
    
    # Component 2: Std matching (generous scoring)  
    std_diff = abs(actual_std - target_std)
    std_tolerance = target_std * 0.8  # 80% tolerance
    std_score = max(0, 1.0 - (std_diff / std_tolerance))
    std_score = std_score ** 0.5  # Square root to boost scores
    
    # Component 3: Shape normality (relaxed)
    try:
        if actual_std > 1e-8:
            # Test shape only by standardizing
            standardized = (solution - actual_mean) / actual_std
            _, p_value = stats.kstest(standardized, 'norm')
            shape_score = min(p_value * 10, 1.0)  # 10x boost instead of 2x
            shape_score = shape_score ** 0.3  # Even more boost for low p-values
        else:
            shape_score = 0.0
    except:
        shape_score = 0.0
    
    # Component 4: Bonus for reasonable range
    reasonable_range = np.sum((np.abs(solution) >= 0.001) & (np.abs(solution) <= 2.0))
    range_bonus = reasonable_range / len(solution)
    
    # Combined score with boosting
    base_score = 0.4 * mean_score + 0.4 * std_score + 0.2 * shape_score
    boosted_score = base_score + 0.2 * range_bonus
    
    # Final boost: square root to lift all scores
    final_score = np.sqrt(boosted_score)
    
    return min(1.0, final_score)
    
def evaluate_kurtosis(solution, target_kurtosis=12.5):
    try:
        solution_kurtosis = stats.kurtosis(solution) + 3
        kurt_score = max(0, 1.0 - abs(solution_kurtosis - target_kurtosis) / 3.0)
        return kurt_score
    except:
        print("Error calculating kurtosis.")
        return 0.0

def evaluate_weight_magnitudes(solution, target_range=(0.01, 1.0)):
    """
    Penalize weights that are too large or too small by rewarding the
    number of weights within a reasonable range.
    """
    min_target, max_target = target_range
    
    reasonable_weights = np.sum((np.abs(solution) >= min_target) & (np.abs(solution) <= max_target))
    total_weights = len(solution)
    
    # Bonus for having most weights in reasonable range
    range_score = reasonable_weights / total_weights
    
    # Penalty for extreme outliers
    extreme_weights = np.sum(np.abs(solution) > max_target * 3)
    outlier_penalty = extreme_weights / total_weights
    return max(0, range_score - outlier_penalty)  
    
def on_gen(ga_instance):
    """
    Callback function to print progress at each generation.
    """
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

def on_fitness(ga_instance, population_fitness):
    print(f"Fitness values: {population_fitness}")
    
def run_ga_optimisation(num_solutions = 10, num_generations = 250, num_parents_mating = 5, mean = 0.0, std_dev = 0.1, kurtosis = 12.5):
    
    global mlp_template

    print("Initializing genetic algorithm population...")
    torch_ga = pygad.torchga.TorchGA(model=mlp_template, num_solutions=num_solutions)
    
    # Debug: Check if initial population is reasonable
    model_weights = torch.tensor(torch_ga.population_weights)
    print(f"Initial population num params: {model_weights.shape}")
    print(f"Initial population stats: min={model_weights.min()}, "
          f"max={model_weights.max()}, "
          f"mean={model_weights.mean()}, ")
    
    initial_weights = pygad.torchga.model_weights_as_vector(model=mlp_template).copy()
    print(f"Weights before GA: {initial_weights[:10]}...")
    
    # Test initial fitness
    initial_fitness = fitness_func(None, torch_ga.population_weights[0], 0)
    print(f"Initial solution fitness: {initial_fitness}")

    initial_population = torch_ga.population_weights

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=on_gen, 
                        save_solutions=False
                    )
                    
    print("Starting PyGAD optimization...")
    ga_instance.run()
    print("\nPyGAD optimization finished.")

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    weights = pygad.torchga.model_weights_as_dict(model=mlp_template,
                                        weights_vector=solution)
    mlp_template.load_state_dict(weights)
    weights, biases = get_stepml_parameters(mlp_template)
    weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)
    
    plot_category_histograms(model_name="StepMLP with GA Optimisation", weights_data=weights_data, biases_data=biases_data, save_path="histograms/stepmlp/ga_optimised_stepmlp_histograms.pdf", custom_format=stepmlp_histogram_format)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GA optimisation on StepMLP")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of rounds for hashing")
    parser.add_argument("--test_phrase", type=str, default="Shht! I am a secret message.", help="Test phrase to hash")
    parser.add_argument("--num_solutions", type=int, default=10, help="Number of solutions for GA")
    parser.add_argument("--num_generations", type=int, default=20, help="Number of generations for GA")
    parser.add_argument("--num_parents_mating", type=int, default=2, help="Number of parents mating for GA")
    args = parser.parse_args()

    print(f"Creating StepMLP from message: {args.test_phrase} with {args.n_rounds} rounds.")
    
    sample_input = const("11101110101")
    sample_output : list[Signal] = [and_gate(sample_input)]

    graph = compile_from_example(inputs=sample_input, outputs=sample_output)
    mlp_template = GACompatibleStepMLP.from_graph(graph)

    weights, biases = get_stepml_parameters(mlp_template)
    weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)

    plot_category_histograms(model_name="StepMLP without GA Optimisation", weights_data=weights_data, biases_data=biases_data, save_path="histograms/stepmlp/before_ga_optimised_stepmlp_histograms.pdf", custom_format=stepmlp_histogram_format)


    input_tensor = torch.tensor([s.activation for s in sample_input], dtype=torch.float64)
    output_tensor = mlp_template(input_tensor)
    print(f"Number of parameters: {len(list(mlp_template.parameters()))} (Total: {sum([p.numel() for p in mlp_template.parameters()])})")
    
    mlp_template, input_tensor, output_tensor = create_gacompatible_stepmlp_from_message(args.test_phrase, n_rounds=args.n_rounds)

    run_ga_optimisation(num_solutions=args.num_solutions,
                        num_generations=args.num_generations,
                        num_parents_mating=args.num_parents_mating)
    