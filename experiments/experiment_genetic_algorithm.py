import torch
import numpy as np
import sys
import pygad
import pygad.torchga
import time
import argparse

from msc_project.analysis.analysis_utils import get_param_stats, get_stepml_parameters, plot_category_histograms
from msc_project.models.ga_compatible_stepml import GACompatibleStepMLP, create_gacompatible_stepmlp_from_message

mlp_template = None
input_tensor = None
output_tensor = None

def fitness_func(ga_instance, solution, solution_idx):

    global input_tensor, output_tensor, torch_ga, mlp_template

    start_time = time.time()
    
    assert input_tensor is not None, "Input tensor is not initialized."
    assert output_tensor is not None, "Output tensor is not initialized."   
    assert mlp_template is not None, "MLP template is not initialized."

    original_weights = pygad.torchga.model_weights_as_vector(mlp_template)

    pygad.torchga.model_weights_as_dict(model=mlp_template, weights_vector=solution)

    with torch.no_grad():
        predicted_output = mlp_template(input_tensor)
        # Objective 1: Correctness
        if not torch.allclose(predicted_output, output_tensor, atol=1e-2):
            return 0.0
        
        fitness = 10.0
        # Objective 2: Robustness to noise
        noise = np.random.normal(0, 0.1, solution.shape)
        noisy_solution = solution + noise
        pygad.torchga.model_weights_as_dict(model=mlp_template, weights_vector=noisy_solution)            
        noisy_output = mlp_template(input_tensor)
        if torch.allclose(noisy_output, output_tensor, atol=1e-4):
            fitness += 5.0
        # Objective 3: Obscurity
        l2_penalty = np.linalg.norm(solution)
        fitness -= 0.1 * l2_penalty  # Apparently need small factor to balance the penalty
        end_time = time.time()
        if solution_idx == 0:  # Print timing for first solution only
            print(f"Fitness evaluation took: {end_time - start_time:.4f} seconds")
        
        # Reset weights 
        pygad.torchga.model_weights_as_dict(model=mlp_template, weights_vector=original_weights)
        return fitness
        
        
    
def on_gen(ga_instance):
    """
    Callback function to print progress at each generation.
    """
    generation = ga_instance.generations_completed
    best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    
    sys.stdout.write(f"Generation {generation:4} | Best Fitness: {best_fitness:8.4f}\r")
    sys.stdout.flush()
    
def run_ga_optimisation(num_solutions = 10, num_generations = 250, num_parents_mating = 5):
    
    global mlp_template

    print("Initializing genetic algorithm population...")
    torch_ga = pygad.torchga.TorchGA(model=mlp_template, num_solutions=num_solutions)
    initial_population = torch_ga.population_weights

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

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    pygad.torchga.model_weights_as_dict(model=mlp_template,
                                        weights_vector=solution)


    weights, biases = get_stepml_parameters(mlp_template)
    weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)

    plot_category_histograms(model_name="StepMLP with GA Optimisation", weights_data=weights_data, biases_data=biases_data, save_path="ga_optimised_stepmlp_histograms.pdf")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GA optimisation on StepMLP")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of rounds for hashing")
    parser.add_argument("--test_phrase", type=str, default="Shht! I am a secret message.", help="Test phrase to hash")
    parser.add_argument("--num_solutions", type=int, default=10, help="Number of solutions for GA")
    parser.add_argument("--num_generations", type=int, default=20, help="Number of generations for GA")
    parser.add_argument("--num_parents_mating", type=int, default=2, help="Number of parents mating for GA")
    args = parser.parse_args()

    print(f"Creating StepMLP from message: {args.test_phrase} with {args.n_rounds} rounds.")
    mlp_template, input_tensor, output_tensor = create_gacompatible_stepmlp_from_message(args.test_phrase, n_rounds=args.n_rounds)

    run_ga_optimisation(num_solutions=args.num_solutions,
                        num_generations=args.num_generations,
                        num_parents_mating=args.num_parents_mating)
    