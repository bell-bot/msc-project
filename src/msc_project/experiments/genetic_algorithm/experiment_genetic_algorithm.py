from dataclasses import asdict
import logging
import random
from typing import cast
import torch
import numpy as np
import pygad
import pygad.torchga
import argparse
from scipy import stats
import copy
from pathlib import Path
from transformers import AutoModelForCausalLM

from circuits.examples.keccak import Keccak
from circuits.utils.format import Bits, format_msg
from msc_project.algorithms.genetic_algorithm.objectives import (
    evaluate_correctness,
    evaluate_distribution_stats,
    evaluate_emd_to_target_dist,
    evaluate_ks_statistic,
    evaluate_normal_distribution,
    evaluate_unique_params,
)

from msc_project.algorithms.genetic_algorithm.utils import GARunConfig
from msc_project.circuits_custom.custom_stepmlp import NPCompatibleStepMLP
from msc_project.evaluation.metrics import earth_movers_distance
from msc_project.utils.experiment_utils import generate_experiment_id, save_experiment_info
from msc_project.utils.experiment_utils import plot_fitness_over_generations
from msc_project.analysis.analysis_utils import (
    get_param_stats,
    get_stepml_parameters,
    plot_category_histograms,
    stepmlp_histogram_format,
)
from msc_project.utils.logging_utils import TimedLogger
from msc_project.utils.model_utils import get_mlp_layers, process_mlp_layers, unfold_stepmlp_parameters
from msc_project.utils.sampling import WeightSampler

EXPERIMENT_RESULTS_DIR = "results/genetic_algorithm_experiments"
EXPERIMENT_TYPE = "genetic_algorithm"

logging.setLoggerClass(TimedLogger)
LOG: TimedLogger = cast(TimedLogger, logging.getLogger(__name__))

def create_batch_fitness_func(mlp_template, input_bits, output_bits, target_weights, target_biases):

    def fitness_func(ga_instance, solutions, solution_indices) -> list[float]:

        local_model = copy.deepcopy(mlp_template)
        batch_fitness : list[float] = []

        for (solution, solution_idx) in zip(solutions, solution_indices):
            solution_weights = pygad.torchga.model_weights_as_dict(
                model=local_model, weights_vector=solution
            )
            local_model.load_state_dict(solution_weights)
            local_model_weights, local_model_biases = unfold_stepmlp_parameters(local_model)
            with torch.no_grad():

                correctness_score = evaluate_correctness(local_model, input_bits, output_bits)

                if correctness_score == 0:
                    batch_fitness.append(0.0)
                    continue

                #dist_stats_weights = evaluate_distribution_stats(local_model_weights, target_mean_weights, target_std_weights, target_kurtosis_weights)
                #dist_stats_biases = evaluate_distribution_stats(local_model_biases, target_mean_biases, target_std_biases, target_kurtosis_biases)
                dist_stats_weights = 0.0
                dist_stats_biases = 0.0
                unique_elems_score = evaluate_unique_params(solution)
                #unique_elems_score = 0.0
                weights_ks_statistic = evaluate_ks_statistic(target_weights, local_model_weights)
                biases_ks_statistic = evaluate_ks_statistic(target_biases, local_model_biases)
                final_score = correctness_score + ( (weights_ks_statistic/2.0) + biases_ks_statistic + dist_stats_weights + dist_stats_biases + unique_elems_score)

                #LOG.info(f"\nSolution {solution_idx}-------\n\tKS Statistic Weights: {1.0-weights_ks_statistic:.4f}\n\tKS Statistic Biases: {1.0-biases_ks_statistic:.4f}\n\tStats Score Weights: {dist_stats_weights:.4f}\n\tStats Score Biases: {dist_stats_biases:.4f}\n\tUnique Elems Score: {unique_elems_score:.4f}\n\t\tTOTAL SCORE = {final_score:.4f}")
                batch_fitness.append(final_score.item())
        return batch_fitness

    return fitness_func
    
def create_fitness_func(mlp_template, input_bits, output_bits, target_weights, target_biases):

    def fitness_func(ga_instance, solution, solution_idx):

        local_model = copy.deepcopy(mlp_template)
        solution_weights = pygad.torchga.model_weights_as_dict(
            model=local_model, weights_vector=solution
        )
        local_model.load_state_dict(solution_weights)
        local_model_weights, local_model_biases = unfold_stepmlp_parameters(local_model)
        with torch.no_grad():

            correctness_score = evaluate_correctness(local_model, input_bits, output_bits)

            if correctness_score == 0:
                return 0.0
            
            #distribution_stats_score = evaluate_distribution_stats(solution)
            unique_elems_score = evaluate_unique_params(solution)
            weights_ks_statistic = evaluate_ks_statistic(target_weights, local_model_weights)
            biases_ks_statistic = evaluate_ks_statistic(target_biases, local_model_biases)
            
            final_score = correctness_score + weights_ks_statistic + biases_ks_statistic + unique_elems_score

            #LOG.info(f"\nSolution {solution_idx}-------\n\tKS Statistic Weights: {1.0-weights_ks_statistic:.4f}\n\tKS Statistic Biases: {1.0-biases_ks_statistic:.4f}\n\tUnique elements score = {unique_elems_score:.4f}\n\t\tTOTAL SCORE = {final_score:.4f}")

            return final_score

    return fitness_func

def on_gen(ga_instance):
    """
    Callback function to print progress at each generation.
    """
    LOG.info(f"Generation = {ga_instance.generations_completed}; Fitness    = {ga_instance.best_solution()[1]}")

def initialise_population_from_target(mlp_template: NPCompatibleStepMLP, num_solutions: int, target_weights: torch.Tensor, target_biases: torch.Tensor) -> pygad.torchga.TorchGA:
    weight_sampler = WeightSampler(target_weights)
    bias_sampler = WeightSampler(target_biases)

    params = []
    for _, layer in mlp_template.named_parameters():
        mlp_bias = layer[0]
        mlp_weight = layer[1:]

        bias_sample = bias_sampler.sample(num_samples = mlp_bias.numel(), sign = "any").reshape(mlp_bias.shape).unsqueeze(dim=0)
        weight_sample = weight_sampler.sample(num_samples=mlp_weight.numel(), sign = "any").reshape(mlp_weight.shape)
        param = torch.cat([torch.mul(mlp_bias,bias_sample), torch.mul(mlp_weight,weight_sample)], dim=0)
        params.append(param)

    mlp_template.load_params(params)
    return pygad.torchga.TorchGA(model=mlp_template, num_solutions=num_solutions)

def run_ga_optimisation(
    mlp_template,
    ga_run_config: GARunConfig,
    input_bits: Bits,
    output_bits: Bits,
    save_path: str = "results/genetic_algorithm_experiments",
    seed: int = 1
):
    target_model = AutoModelForCausalLM.from_pretrained(ga_run_config.target_model_name)
    mlp_layers = get_mlp_layers(target_model)
    gpt2_weights, gpt2_biases = process_mlp_layers(mlp_layers, 1.0)
    
    ga_config = ga_run_config.ga_config
    print("Initializing genetic algorithm population...")
    torch_ga = pygad.torchga.TorchGA(model=mlp_template, num_solutions=ga_config.sol_per_pop)
    # model_weights = np.array(torch_ga.population_weights).flatten()
    # print(f"Initial population num params: {model_weights.shape}")
    # print(
    #     f"Initial population stats: min={model_weights.min()}, "
    #     f"max={model_weights.max()}, "
    #     f"mean={model_weights.mean()}, "
    # )

    # initial_population = []
    # # diversify initial population
    # for i in range(0, num_solutions):
    #     print(f"Generated solution candidate {i+1}.")
    #     population_weights = np.random.random(model_weights.shape)
    #     initial_population.append(population_weights)

    # print("Initializing genetic algorithm population...")
    # torch_ga = pygad.torchga.TorchGA(model=mlp_template, num_solutions=num_solutions)
    
    # model_weights = torch.tensor(np.array(torch_ga.population_weights))
    # print(f"Initial population num params: {model_weights.shape}")
    # print(f"Initial population stats: min={model_weights.min()}, "
    #       f"max={model_weights.max()}, "
    #       f"mean={model_weights.mean()}, ")
    
    # initial_weights = pygad.torchga.model_weights_as_vector(model=mlp_template).copy()
    # print(f"Weights before GA: {initial_weights[:10]}...")
    
    fitness_func = ga_run_config.create_fitness_func(mlp_template, input_bits, output_bits, gpt2_weights, gpt2_biases)
    ga_config.fitness_func = fitness_func
    # initial_fitness = fitness_func(None, torch_ga.population_weights[0], 0)
    # print(f"Initial solution fitness: {initial_fitness}")

    ga_config.initial_population = torch_ga.population_weights
    ga_config.random_seed = seed
    ga_config.on_generation = on_gen
    ga_config.logger = LOG
    # ga_instance = pygad.GA(
    #     num_generations=num_generations,
    #     num_parents_mating=num_parents_mating,
    #     initial_population=initial_population,
    #     fitness_func=fitness_func,
    #     on_generation=on_gen,
    #     save_solutions=False,
    #     random_seed=seed,
    #     fitness_batch_size=10,
    #     parent_selection_type="tournament",
    # )


    ga_instance = pygad.GA(**asdict(ga_config))
    summary_file = open(f"{save_path}/summary.txt", "w")
    

    print("Starting PyGAD optimization...")
    ga_instance.run()
    print("\nPyGAD optimization finished.")
    summary_file.write(ga_instance.summary())
    summary_file.close()
    plot_fitness_over_generations(ga_instance, save_path)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    weights = pygad.torchga.model_weights_as_dict(model=mlp_template, weights_vector=solution)
    mlp_template.load_state_dict(weights)

    # Verify the output of the GA-optimised model
    verify_ga_optimised_stepml(mlp_template, input_bits, output_bits)

    if save_path:
        torch.save(mlp_template.state_dict(), f"{save_path}/ga_optimised_stepml_model.pth")
        weights, biases = unfold_stepmlp_parameters(mlp_template)
        weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)

        ga_optimised_histogram_save_path = f"{save_path}/ga_optimised_stepml_histograms.pdf"
        plot_category_histograms(
            model_name="StepMLP with GA Optimisation",
            weights_data=weights_data,
            biases_data=biases_data,
            save_path=ga_optimised_histogram_save_path,
            custom_format=stepmlp_histogram_format,
        )

def verify_ga_optimised_stepml(mlp_template, formatted_message: Bits, expected_output: Bits):
    actual_output = mlp_template.infer_bits(formatted_message)
    actual_output_hex = actual_output.bitstr
    print(f"Expected output: {expected_output.bitstr}")
    print(f"Actual output: {actual_output_hex}")

    if actual_output_hex == expected_output.bitstr:
        print("GA-optimised StepMLP produces the expected output.")
    else:
        print("GA-optimised StepMLP does NOT produce the expected output.")
        raise ValueError("Output mismatch after GA optimisation.")

def run_ga(ga_run_config: GARunConfig):

    experiment_save_path = f"{EXPERIMENT_RESULTS_DIR}/{ga_run_config.experiment_name}"
    Path(experiment_save_path).mkdir(parents=True, exist_ok=True)

    LOG.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{experiment_save_path}/experiment.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)
    LOG.handlers = [file_handler]
    with open(f"{experiment_save_path}/info.txt", "a") as f:
        f.write(str(ga_run_config))
        f.close()

    start_i = ga_run_config.run_start_idx

    for i in range(start_i, ga_run_config.num_experiments + start_i):
        LOG.info(f"Experiment {i+1} ---------------- ")
        
        save_path = f"{EXPERIMENT_RESULTS_DIR}/{ga_run_config.experiment_name}/run_{i+1}"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        seed = i
        keccak = Keccak(c=20, log_w=1, n=3)
        trigger_bits = format_msg(ga_run_config.test_trigger, keccak.msg_len)
        payload_bits = format_msg(ga_run_config.test_payload, keccak.d)

        mlp_template = NPCompatibleStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist, payload=payload_bits.bitlist, k=keccak
        )

        run_ga_optimisation(mlp_template, ga_run_config, trigger_bits, payload_bits, save_path, seed)
        LOG.info(f" DONE \n")