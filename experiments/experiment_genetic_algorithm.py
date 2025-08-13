import logging
import torch
import numpy as np
import pygad
import pygad.torchga
import argparse
from scipy import stats
import copy
from pathlib import Path

from circuits.examples.keccak import Keccak
from circuits.examples.simple_example import and_gate
from circuits.neurons.core import Bit
from circuits.sparse.compile import compiled_from_io
from circuits.utils.format import Bits, format_msg
from msc_project.algorithms.genetic_algorithm.objectives import evaluate_correctness, evaluate_distribution_stats, evaluate_normal_distribution

from msc_project.circuits_custom.custom_stepmlp import GACompatibleStepMLP
from utils import generate_experiment_id, plot_fitness_over_generations, save_experiment_info
from msc_project.analysis.analysis_utils import get_param_stats, get_stepml_parameters, plot_category_histograms, stepmlp_histogram_format

EXPERIMENT_RESULTS_DIR = "results/genetic_algorithm_experiments"
EXPERIMENT_TYPE = "genetic_algorithm"

def create_fitness_func(mlp_template, input_bits, output_bits):

    def fitness_func(ga_instance, solution, solution_idx):

        local_model = copy.deepcopy(mlp_template)
        solution_weights = pygad.torchga.model_weights_as_dict(model=local_model, weights_vector=solution)
        local_model.load_state_dict(solution_weights)
        with torch.no_grad():
            
            correctness_score = evaluate_correctness(local_model, input_bits, output_bits)
            distribution_stats_score = evaluate_distribution_stats(solution)

            final_score = correctness_score + distribution_stats_score
            
            return final_score
        
    return fitness_func
    
def on_gen(ga_instance):
    """
    Callback function to print progress at each generation.
    """
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

def on_fitness(ga_instance, population_fitness):
    print(f"Fitness values: {population_fitness}")
    
def run_ga_optimisation(mlp_template, input_bits, output_bits, num_solutions = 10, num_generations = 250, num_parents_mating = 5, mean = 0.0, std_dev = 0.1, kurtosis = 12.5,  save_path : str | None="results/genetic_algorithm_experiments"):
    
    print("Initializing genetic algorithm population...")
    torch_ga = pygad.torchga.TorchGA(model=mlp_template, num_solutions=num_solutions)
    
    model_weights = torch.tensor(np.array(torch_ga.population_weights))
    print(f"Initial population num params: {model_weights.shape}")
    print(f"Initial population stats: min={model_weights.min()}, "
          f"max={model_weights.max()}, "
          f"mean={model_weights.mean()}, ")

    initial_population = torch_ga.population_weights
    # diversify initial population
    for i in range(1, num_solutions):
        noise = np.random.normal(0, 0.001, initial_population[i].shape)
        initial_population[i] += noise

    fitness_func = create_fitness_func(mlp_template, input_bits, output_bits)

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=on_gen, 
                        save_solutions=False,
                    )
                    
    print("Starting PyGAD optimization...")
    ga_instance.run()
    print("\nPyGAD optimization finished.")

    if save_path:
        plot_fitness_over_generations(ga_instance, save_path)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    weights = pygad.torchga.model_weights_as_dict(model=mlp_template,
                                        weights_vector=solution)
    mlp_template.load_state_dict(weights)

    # Verify the output of the GA-optimised model
    verify_ga_optimised_stepml(mlp_template, input_bits, output_bits)

    if save_path:
        torch.save(mlp_template.state_dict(), f"{save_path}/ga_optimised_stepml_model.pth")
        weights, biases = get_stepml_parameters(mlp_template)
        weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)
        
        ga_optimised_histogram_save_path = f"{save_path}/ga_optimised_stepml_histograms.pdf"
        plot_category_histograms(model_name="StepMLP with GA Optimisation", weights_data=weights_data, biases_data=biases_data, save_path=ga_optimised_histogram_save_path, custom_format=stepmlp_histogram_format)

def verify_ga_optimised_stepml(mlp_template, formatted_message: Bits, expected_output: Bits):
    actual_output = mlp_template.infer_bits(formatted_message)
    actual_output_hex = actual_output.hex
    print(f"Expected output: {expected_output.hex}")
    print(f"Actual output: {actual_output_hex}")
    
    if actual_output_hex == expected_output.hex:
        print("GA-optimised StepMLP produces the expected output.")
    else:
        print("GA-optimised StepMLP does NOT produce the expected output.")
        raise ValueError("Output mismatch after GA optimisation.")
    
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

    keccak = Keccak(c=20, l=1, n=3)
    trigger_bits = format_msg(args.test_phrase, keccak.msg_len)
    payload_bits = format_msg(args.payload, keccak.d)

    mlp_template = GACompatibleStepMLP.create_with_backdoor(trigger=trigger_bits.bitlist, payload=payload_bits.bitlist, k=keccak)
    print(f"Created StepMLP with {mlp_template.n_params} parameters.")


    run_ga_optimisation(mlp_template=mlp_template,
                        input_bits=trigger_bits,
                        output_bits=payload_bits,
                        num_solutions=args.num_solutions,
                        num_generations=args.num_generations,
                        num_parents_mating=args.num_parents_mating,
                        save_path=save_path)
    