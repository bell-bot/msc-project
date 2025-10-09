from collections.abc import Callable
import copy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pygad
import pygad.torchga
from scipy import stats
import torch
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import Bits
from msc_project.algorithms.genetic_algorithm.objectives import evaluate_correctness, evaluate_distribution_stats, evaluate_ks_statistic, evaluate_unique_params
from msc_project.utils.logging_utils import TimedLogger
from msc_project.utils.model_utils import unfold_stepmlp_parameters


def verify_ga_optimised_stepml(mlp_template, formatted_message: Bits, expected_output: Bits):
    actual_output = mlp_template.infer_bits(formatted_message)
    actual_output_text = actual_output.bitstr
    print(f"Expected output: {expected_output.bitstr}")
    print(f"Actual output: {actual_output_text}")

    if actual_output_text == expected_output.bitstr:
        print("GA-optimised StepMLP produces the expected output.")
    else:
        print("GA-optimised StepMLP does NOT produce the expected output.")
        raise ValueError("Output mismatch after GA optimisation.")

@dataclass
class GAConfig:
    num_generations : int
    num_parents_mating: int
    fitness_func: Callable[[pygad.GA, list[Any], int | list[Any]], Any] = None
    fitness_batch_size : int = None
    initial_population : list[Any] = None
    sol_per_pop : int = None
    num_genes=None
    init_range_low=-4
    init_range_high=4
    gene_type=float
    parent_selection_type : str ="sss"
    keep_parents=-1
    keep_elitism=1
    K_tournament=3
    crossover_type="single_point"
    crossover_probability=None
    mutation_type="random"
    mutation_probability=None
    mutation_by_replacement=False
    mutation_percent_genes='default'
    mutation_num_genes=None
    random_mutation_min_val=-1.0
    random_mutation_max_val=1.0
    gene_space=None
    gene_constraint=None
    sample_size=100
    allow_duplicate_genes=True
    on_start=None
    on_fitness=None
    on_parents=None
    on_crossover=None
    on_mutation=None
    on_generation : Callable =None
    on_stop=None
    save_best_solutions=False
    save_solutions : bool =False
    suppress_warnings=False
    stop_criteria=None
    parallel_processing : int = None
    random_seed: int = None
    logger : TimedLogger =None

    def __str__(self):
        s = ""
        for field in fields(self):
            s += f"{field.name}: {getattr(self, field.name)}\n"
    
        return s

@dataclass
class GARunConfig:
    ga_config: GAConfig
    num_experiments: int
    create_fitness_func: Callable[[StepMLP, Bits, Bits, torch.Tensor, torch.Tensor, float, float], Callable[[pygad.GA, list[Any], int | list[Any]], Any]]
    target_model_name: str
    experiment_name: str
    factor_w: float = 1.0
    factor_b: float = 1.0
    test_trigger : str = "Test"
    test_payload : str = "tseT"
    logger : TimedLogger = None
    run_start_idx : int = 0

    def __str__(self):
        s = ""
        for field in fields(self):
            s += f"{field.name}: {getattr(self, field.name)}\n"
    
        return s

def create_simple_fitness_func(mlp_template, input_bits, output_bits, target_weights, target_biases, factor_w = 1.0, factor_b = 1.0):
    
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

                unique_elems_score = evaluate_unique_params(solution)
                weights_ks_statistic = evaluate_ks_statistic(target_weights, local_model_weights)
                biases_ks_statistic = evaluate_ks_statistic(target_biases, local_model_biases)
                final_score = correctness_score + ( factor_w*weights_ks_statistic + factor_b*biases_ks_statistic + unique_elems_score)

                batch_fitness.append(final_score.item())
        return batch_fitness

    return fitness_func

def create_distribution_aware_fitness_func(mlp_template, input_bits, output_bits, target_weights, target_biases,  factor_w = 1.0, factor_b = 1.0):

    target_weights_mean = target_weights.mean()
    target_weights_std = target_weights.std()
    target_weights_kurtosis = stats.kurtosis(target_weights)
    target_biases_mean = target_biases.mean()
    target_biases_std = target_biases.std()
    target_biases_kurtosis = stats.kurtosis(target_biases)

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
                
                dist_stats_weights = evaluate_distribution_stats(local_model_weights, target_weights_mean, target_weights_std, target_weights_kurtosis)
                dist_stats_biases = evaluate_distribution_stats(local_model_biases, target_biases_mean, target_biases_std, target_biases_kurtosis)
                unique_elems_score = evaluate_unique_params(solution)
                weights_ks_statistic = evaluate_ks_statistic(target_weights, local_model_weights)
                biases_ks_statistic = evaluate_ks_statistic(target_biases, local_model_biases)
                final_score = correctness_score + factor_w * (weights_ks_statistic + dist_stats_weights) + factor_b * (biases_ks_statistic + dist_stats_biases) + unique_elems_score

                batch_fitness.append(final_score.item())
        return batch_fitness

    return fitness_func

def create_magnitude_aware_fitness_func(mlp_template, input_bits, output_bits, target_weights, target_biases,  factor_w = 1.0, factor_b = 1.0):

    target_weights_mean = target_weights.mean()
    target_weights_std = target_weights.std()
    target_weights_kurtosis = stats.kurtosis(target_weights)
    target_biases_mean = target_biases.mean()
    target_biases_std = target_biases.std()
    target_biases_kurtosis = stats.kurtosis(target_biases)

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
                
                dist_stats_weights = evaluate_distribution_stats(local_model_weights, target_weights_mean, target_weights_std, target_weights_kurtosis)
                dist_stats_biases = evaluate_distribution_stats(local_model_biases, target_biases_mean, target_biases_std, target_biases_kurtosis)
                unique_elems_score = evaluate_unique_params(solution)
                weights_ks_statistic = evaluate_ks_statistic(target_weights, local_model_weights)
                biases_ks_statistic = evaluate_ks_statistic(target_biases, local_model_biases)
                magnitude_score = 1.0 / (np.mean(np.absolute(solution)))
                final_score = correctness_score + factor_w * (weights_ks_statistic + dist_stats_weights) + factor_b * (biases_ks_statistic + dist_stats_biases) + unique_elems_score + magnitude_score

                batch_fitness.append(final_score.item())
        return batch_fitness

    return fitness_func