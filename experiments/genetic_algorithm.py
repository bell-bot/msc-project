import argparse
import copy
from pathlib import Path

import numpy as np
from circuits.examples.keccak import Keccak
from circuits.utils.format import Bits, format_msg
from circuits.dense.mlp import StepMLP
import torch
import logging
import pygad

from msc_project.circuits_custom.custom_stepmlp import GACompatibleStepMLP
from .utils import generate_experiment_id, plot_fitness_over_generations, save_experiment_info
from msc_project.algorithms.genetic_algorithm.objectives import evaluate_normal_distribution
from msc_project.algorithms.genetic_algorithm.utils import verify_ga_optimised_stepml
from msc_project.analysis.analysis_utils import get_param_stats, get_stepml_parameters, plot_category_histograms, stepmlp_histogram_format

from scipy.spatial.distance import hamming

logger = logging.getLogger(__name__)
EXPERIMENT_RESULTS_DIR = "results/genetic_algorithm_experiments"
EXPERIMENT_TYPE = "genetic_algorithm"

def create_fitness_func_layer(mlp_template: StepMLP, trigger_bits: Bits, payload_bits: Bits, target_layer_name: str):
    """
    Create a fitness function that optimises a single layer of an MLP.
    """

    original_dtype = mlp_template.state_dict()[target_layer_name].dtype

    def fitness_func(ga_instance, solution, solution_idx):

        local_model = copy.deepcopy(mlp_template)
        state_dict = local_model.state_dict()

        # Update the weights of the specified target layer
        target_layer_shape = state_dict[target_layer_name].shape
        solution_tensor = torch.tensor(solution.reshape(target_layer_shape), dtype=original_dtype)
        state_dict[target_layer_name] = solution_tensor
        local_model.load_state_dict(state_dict)

        with torch.no_grad():

            predicted_output = local_model.infer_bits(trigger_bits).bitlist

            # Objective 1: Preserve the backdoor functionality
            correctness_score = 1.0 - hamming(np.array(payload_bits.bitlist), np.array(predicted_output)) 
            fitness = correctness_score

            # Objective 2: Mimic normal distribution
            fitness += evaluate_normal_distribution(solution)

        return fitness
    
    return fitness_func

def initialise_population(mlp_template: StepMLP, num_solutions: int, target_layer_name: str):
    """
    Initialise a population of solutions for the genetic algorithm.
    Each solution is a flattened tensor of the weights of the target layer.
    """

    initial_layer_weights = mlp_template.state_dict()[target_layer_name].numpy().flatten()
    initial_population = np.array([initial_layer_weights.copy() for _ in range(num_solutions)])

    # Add some noise to initial population to ensure diversity
    for i in range(initial_population.shape[0]):
        initial_population[i, :] += np.random.normal(0.0, 0.001, initial_population[i, :].shape)

    return initial_population

def on_generation(ga_instance):
    """
    Callback function to log the best fitness and solution at each generation.
    """
    best_solution, best_fitness, _ = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness}")
    print(f"Best Solution: {best_solution}")
    
def run_ga_optimisation_on_layer(mlp_template: StepMLP, trigger_bits: Bits, payload_bits: Bits, target_layer_name: str, num_solutions=50, num_generations=100, num_parents_mating=2, save_path=None):
    """
    Run a genetic algorithm to optimise a single layer of an MLP.
    """

    print(f"Running GA optimisation on layer: {target_layer_name}")
    print("Initialising population...")
    initial_population = initialise_population(mlp_template, num_solutions, target_layer_name)

    fitness_func = create_fitness_func_layer(mlp_template, trigger_bits, payload_bits, target_layer_name)
    
    print("Creating GA instance...")
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           initial_population=initial_population,
                           crossover_probability=0.5,
                           mutation_probability=0.2,
                           on_generation=on_generation)
    
    print("Starting GA optimisation...")
    ga_instance.run()
    print("GA optimisation completed.")

    if save_path:
        plot_fitness_over_generations(ga_instance, save_path)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    state_dict = mlp_template.state_dict()
    target_layer_shape = state_dict[target_layer_name].shape
    state_dict[target_layer_name] = torch.tensor(solution.reshape(target_layer_shape))
    mlp_template.load_state_dict(state_dict)

    # Verify the output of the GA-optimised model
    verify_ga_optimised_stepml(mlp_template, trigger_bits, payload_bits)

    if save_path:
        torch.save(mlp_template.state_dict(), f"{save_path}/ga_optimised_stepml_model.pth")
        weights, biases = get_stepml_parameters(mlp_template)
        weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)
        
        ga_optimised_histogram_save_path = f"{save_path}/ga_optimised_stepml_histograms.pdf"
        plot_category_histograms(model_name="StepMLP with GA Optimisation", weights_data=weights_data, biases_data=biases_data, save_path=ga_optimised_histogram_save_path, custom_format=stepmlp_histogram_format)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GA optimisation on StepMLP")
    parser.add_argument("--test_phrase", type=str, default="Shht! I am a secret message.", help="Test phrase to hash")
    parser.add_argument("--payload", type=str, default="Consider yourself BACKDOORED", help="Payload for the backdoor")
    parser.add_argument("--num_solutions", type=int, default=100, help="Number of solutions for GA")
    parser.add_argument("--num_generations", type=int, default=20, help="Number of generations for GA")
    parser.add_argument("--num_parents_mating", type=int, default=10, help="Number of parents mating for GA")
    parser.add_argument("--save", type=bool, default=True, help="Save experiment results")
    args = parser.parse_args()

    print(f"Creating StepMLP from message: {args.test_phrase}")

    experiment_id = generate_experiment_id(EXPERIMENT_TYPE)
    save_path = f"{EXPERIMENT_RESULTS_DIR}/{experiment_id}" if args.save else None

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

        save_experiment_info(experiment_id, vars(args), save_path)
        logging.basicConfig(filename=f"{save_path}/experiment.log", format='%(asctime)s - %(levelname)s - %(message)s')

    keccak = Keccak()
    trigger_bits = format_msg(args.test_phrase, keccak.msg_len)
    payload_bits = format_msg(args.payload, keccak.d)

    mlp_template = GACompatibleStepMLP.create_with_backdoor(trigger=trigger_bits.bitlist, payload=payload_bits.bitlist, k=keccak)
    print(f"Created StepMLP with {mlp_template.n_params} parameters.")

    last_layer_name = list(mlp_template.state_dict().keys())[-1]
    print(f"Last layer name: {last_layer_name}; shape: {mlp_template.state_dict()[last_layer_name].shape}")

    # Run GA optimisation on the last layer
    run_ga_optimisation_on_layer(mlp_template, trigger_bits, payload_bits, last_layer_name, num_solutions=args.num_solutions, num_generations=args.num_generations, num_parents_mating=args.num_parents_mating, save_path=save_path)
