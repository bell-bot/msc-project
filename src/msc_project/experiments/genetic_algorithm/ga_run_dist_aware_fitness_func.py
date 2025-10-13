import logging
from typing import cast
from experiments.experiment_genetic_algorithm import run_ga
from msc_project.algorithms.genetic_algorithm.utils import GAConfig, GARunConfig, create_distribution_aware_fitness_func, create_simple_fitness_func
from msc_project.utils.logging_utils import TimedLogger

logging.setLoggerClass(TimedLogger)
LOG: TimedLogger = cast(TimedLogger, logging.getLogger(__name__))

ga_config = GAConfig(
    num_generations=50, 
    num_parents_mating=10, 
    sol_per_pop=100, 
    save_solutions=False, 
    fitness_batch_size=10, 
    parent_selection_type="tournament",
    parallel_processing=2,
    )

ga_run_config = GARunConfig(
    ga_config=ga_config,
    num_experiments = 5,
    create_fitness_func = create_distribution_aware_fitness_func,
    target_model_name="gpt2",
    experiment_name="distribution_aware_fitness_func",
    run_start_idx = 5
)

run_ga(ga_run_config)