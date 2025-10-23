import copy
import json
import logging
import os
from typing import cast
import pandas as pd
import torch
from tqdm import tqdm
from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
from msc_project.experiments.fault_tolerant_boolean_circuits.monitor import Monitor
from msc_project.experiments.fault_tolerant_boolean_circuits.perturbable_stepmlp import (
    ParallelMajorityVotingStepMLP,
    PerturbableStepMLP,
)
from msc_project.experiments.majority_voting_backdoor.utils import noise_stepmlp
from msc_project.experiments.fault_tolerant_boolean_circuits.robust_keccak import (
    MultiplexedKeccak,
    RobustKeccak,
)
from msc_project.experiments.fault_tolerant_boolean_circuits.robust_xor_stepmlp import (
    RobustXorMajorityVotingStepMLP,
    RobustXorStepMLP,
)
from msc_project.utils.experiment_utils import RobustnessExperimentSpecs, setup_logging
from msc_project.utils.logging_utils import TimedLogger
from msc_project.utils.model_utils import verify_stepmlp
import numpy as np

logging.setLoggerClass(TimedLogger)
LOG: TimedLogger = cast(TimedLogger, logging.getLogger(__name__))


def run_iteration(mlp_template, trigger_bits, payload_bits, n_samples, std, i):

    n_correct = 0

    for j in tqdm(range(n_samples), f"{i}. Std = {std:.4f}"):
        mlp = copy.deepcopy(mlp_template)

        init_weights = list(mlp.parameters())[0].flatten().detach().to(torch.float32).numpy()[:10]

        noise_stepmlp(mlp, std)
        noised_weights = list(mlp.parameters())[0].flatten().detach().to(torch.float32).numpy()[:10]
        n_correct += verify_stepmlp(mlp, trigger_bits, payload_bits)
        predicted_output = mlp.infer_bits(trigger_bits)

        LOG.info(
            f"\tModel %d\n\tInitial weights:\t[%s]\n\tNoised weights: \t[%s]\n\tExpected output: \t%s\n\tActual output:  \t%s",
            j,
            ", ".join([str(weight) for weight in init_weights]),
            ", ".join([str(weight) for weight in noised_weights]),
            payload_bits.bitstr,
            predicted_output.bitstr,
        )

    return n_correct / n_samples


def run_iteration_with_logging(
    mlp: PerturbableStepMLP,
    trigger_bits,
    payload_bits,
    n_samples,
    std,
    i,
    model_path: str | None = None,
):

    n_correct = 0
    model_file = lambda x: f"{model_path}/models/model_{x}"

    for j in tqdm(range(n_samples), f"{i}. Std = {std:.4f}"):

        init_weights = list(mlp.parameters())[0].flatten().detach().to(torch.float32).numpy()[:10]

        mlp.perturb(std)
        noised_weights = list(mlp.parameters())[0].flatten().detach().to(torch.float32).numpy()[:10]
        predicted_output = mlp.infer_bits(trigger_bits)
        n_correct += predicted_output.bitstr == payload_bits.bitstr

        LOG.info(
            f"\tModel %d\n\tInitial weights:\t[%s]\n\tNoised weights: \t[%s]\n\tExpected output: \t%s\n\tActual output:  \t%s",
            j,
            ", ".join([str(weight) for weight in init_weights]),
            ", ".join([str(weight) for weight in noised_weights]),
            payload_bits.bitstr,
            predicted_output.bitstr,
        )

        if model_path != None:
            p = model_file(j)
            torch.save(mlp.state_dict(), p)

        mlp.reset()

    return n_correct / n_samples


def run_robust_xor(config: RobustnessExperimentSpecs):

    torch.manual_seed(config.random_seed)

    experiment_dir = f"results/robust_xor_backdoor/{config.experiment_name}"
    os.makedirs(os.path.dirname(f"{experiment_dir}/experiment.log"), exist_ok=True)

    # Set up logging
    LOG.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{experiment_dir}/experiment.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)

    LOG.handlers = [file_handler]

    # Save experiment info
    experiment_info_file = open(f"{experiment_dir}/info.txt", "w")
    experiment_info_file.write(json.dumps(config.dict()))

    # Setup the backdoored model. We will initialise it once and then create a copy
    # in each experiment iteration
    with LOG.time("Creating backdoored model template", show_pbar=False):
        keccak = RobustKeccak(c=config.c, log_w=config.log_w, n=config.n)
        trigger_bits = format_msg(config.trigger_str, keccak.msg_len)
        payload_bits = format_msg(config.payload_str, keccak.d)
        mlp_template = RobustXorStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist, payload=payload_bits.bitlist, k=keccak
        )

    results = {}

    for i, standard_deviation in enumerate(config.noise_stds):
        LOG.info(
            f"\nExperiment {i+1} - Standard deviation: {standard_deviation} ---------------------------------\n"
        )

        preserve_rate = run_iteration(
            mlp_template, trigger_bits, payload_bits, config.num_samples, standard_deviation, i
        )
        results[standard_deviation] = preserve_rate

    results_file = open(f"{experiment_dir}/results.json", "w")
    results_file.write(json.dumps(results))


def run_majority_voting(config: RobustnessExperimentSpecs):

    torch.manual_seed(config.random_seed)

    experiment_dir = f"results/robust_xor_backdoor/{config.experiment_name}"
    setup_logging(LOG, experiment_dir)

    # Save experiment info
    experiment_info_file = open(f"{experiment_dir}/info.txt", "w")
    experiment_info_file.write(json.dumps(config.dict()))

    # Setup the backdoored model. We will initialise it once and then create a copy
    # in each experiment iteration
    with LOG.time("Creating backdoored model template", show_pbar=False):
        keccak = RobustKeccak(c=config.c, log_w=config.log_w, n=config.n)
        trigger_bits = format_msg(config.trigger_str, keccak.msg_len)
        payload_bits = format_msg(config.payload_str, keccak.d)
        mlp_template = RobustXorMajorityVotingStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist,
            payload=payload_bits.bitlist,
            k=keccak,
            redundancy=config.redundancy,
        )

    results = {}

    for i, standard_deviation in enumerate(config.noise_stds):
        LOG.info(
            f"\nExperiment {i+1} - Standard deviation: {standard_deviation} ---------------------------------\n"
        )

        preserve_rate = run_iteration(
            mlp_template, trigger_bits, payload_bits, config.num_samples, standard_deviation, i
        )
        results[standard_deviation] = preserve_rate

    results_file = open(f"{experiment_dir}/results.json", "w")
    results_file.write(json.dumps(results))


def run_robust_xor_with_logging(config: RobustnessExperimentSpecs):

    torch.manual_seed(config.random_seed)

    experiment_dir = f"results/experiment_noise_tolerance/{config.experiment_name}"
    setup_logging(LOG, experiment_dir)

    # Save experiment info
    experiment_info_file = open(f"{experiment_dir}/info.txt", "w")
    experiment_info_file.write(json.dumps(config.dict()))

    # Setup the backdoored model. We will initialise it once and then create a copy
    # in each experiment iteration
    with LOG.time("Creating backdoored model template", show_pbar=False):
        keccak = RobustKeccak(c=config.c, log_w=config.log_w, n=config.n)
        trigger_bits = format_msg(config.trigger_str, keccak.msg_len)
        payload_bits = format_msg(config.payload_str, keccak.d)
        mlp_template = PerturbableStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist,
            payload=payload_bits.bitlist,
            k=keccak,
            backdoor_type=config.backdoor_type,
            redundancy=config.redundancy,
        )

    activations = []

    with LOG.time("Saving baseline activations", show_pbar=False):
        monitor = Monitor()
        monitor.register_hooks(mlp_template)
        mlp_template.infer_bits(trigger_bits)
        baseline_activations = monitor.to_dataframe(std=0.0)
        activations.append(baseline_activations)
        monitor.clear_data()

    results = {}
    os.makedirs(os.path.dirname(f"{experiment_dir}/models/"), exist_ok=True)
    for i, standard_deviation in enumerate(config.noise_stds):
        LOG.info(
            f"\nExperiment {i+1} - Standard deviation: {standard_deviation} ---------------------------------\n"
        )

        preserve_rate = run_iteration_with_logging(
            mlp_template, trigger_bits, payload_bits, config.num_samples, standard_deviation, i
        )
        result_activations = monitor.to_dataframe(std=standard_deviation)

        activations.append(result_activations)

        monitor.clear_data()
        results[standard_deviation] = preserve_rate

    activations_df = pd.concat(activations, axis=0)
    activations_df.to_pickle(f"{experiment_dir}/activations.pkl")

    results_file = open(f"{experiment_dir}/results.json", "w")
    results_file.write(json.dumps(results))


def run_multiplexed_xor_with_logging(config: RobustnessExperimentSpecs):

    torch.manual_seed(config.random_seed)

    experiment_dir = f"results/experiment_noise_tolerance/{config.experiment_name}"
    setup_logging(LOG, experiment_dir)

    # Save experiment info
    experiment_info_file = open(f"{experiment_dir}/info.txt", "w")
    experiment_info_file.write(json.dumps(config.dict()))

    # Setup the backdoored model. We will initialise it once and then create a copy
    # in each experiment iteration
    with LOG.time("Creating backdoored model template", show_pbar=False):
        keccak = MultiplexedKeccak(c=config.c, log_w=config.log_w, n=config.n)
        trigger_bits = format_msg(config.trigger_str, keccak.msg_len)
        payload_bits = format_msg(config.payload_str, keccak.d)
        mlp_template = PerturbableStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist,
            payload=payload_bits.bitlist,
            k=keccak,
            backdoor_type=config.backdoor_type,
        )

    activations = []

    with LOG.time("Saving baseline activations", show_pbar=False):
        monitor = Monitor()
        monitor.register_hooks(mlp_template)
        mlp_template.infer_bits(trigger_bits)
        baseline_activations = monitor.to_dataframe(std=0.0)
        activations.append(baseline_activations)
        monitor.clear_data()

    results = {}
    os.makedirs(os.path.dirname(f"{experiment_dir}/models/"), exist_ok=True)
    for i, standard_deviation in enumerate(config.noise_stds):
        LOG.info(
            f"\nExperiment {i+1} - Standard deviation: {standard_deviation} ---------------------------------\n"
        )

        preserve_rate = run_iteration_with_logging(
            mlp_template, trigger_bits, payload_bits, config.num_samples, standard_deviation, i
        )
        result_activations = monitor.to_dataframe(std=standard_deviation)

        activations.append(result_activations)

        monitor.clear_data()
        results[standard_deviation] = preserve_rate

    activations_df = pd.concat(activations, axis=0)
    activations_df.to_pickle(f"{experiment_dir}/activations.pkl")

    results_file = open(f"{experiment_dir}/results.json", "w")
    results_file.write(json.dumps(results))


def run_baseline_with_logging(config: RobustnessExperimentSpecs):

    torch.manual_seed(config.random_seed)

    experiment_dir = f"results/experiment_noise_tolerance/{config.experiment_name}"
    setup_logging(LOG, experiment_dir)

    # Save experiment info
    experiment_info_file = open(f"{experiment_dir}/info.txt", "w")
    experiment_info_file.write(json.dumps(config.dict()))

    # Setup the backdoored model. We will initialise it once and then create a copy
    # in each experiment iteration
    with LOG.time("Creating backdoored model template", show_pbar=False):
        keccak = Keccak(c=config.c, log_w=config.log_w, n=config.n)
        trigger_bits = format_msg(config.trigger_str, keccak.msg_len)
        payload_bits = format_msg(config.payload_str, keccak.d)
        mlp_template = PerturbableStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist,
            payload=payload_bits.bitlist,
            k=keccak,
            backdoor_type=config.backdoor_type,
            redundancy=config.redundancy,
        )

    activations = []

    with LOG.time("Saving baseline activations", show_pbar=False):
        monitor = Monitor()
        monitor.register_hooks(mlp_template)
        mlp_template.infer_bits(trigger_bits)
        baseline_activations = monitor.to_dataframe(std=0.0)
        activations.append(baseline_activations)
        monitor.clear_data()

    results = {}
    os.makedirs(os.path.dirname(f"{experiment_dir}/models/"), exist_ok=True)
    for i, standard_deviation in enumerate(config.noise_stds):
        LOG.info(
            f"\nExperiment {i+1} - Standard deviation: {standard_deviation} ---------------------------------\n"
        )

        preserve_rate = run_iteration_with_logging(
            mlp_template, trigger_bits, payload_bits, config.num_samples, standard_deviation, i
        )
        result_activations = monitor.to_dataframe(std=standard_deviation)

        activations.append(result_activations)

        monitor.clear_data()
        results[standard_deviation] = preserve_rate

    activations_df = pd.concat(activations, axis=0)
    activations_df.to_pickle(f"{experiment_dir}/activations.pkl")

    results_file = open(f"{experiment_dir}/results.json", "w")
    results_file.write(json.dumps(results))


def run_parallel_majority_voting_with_logging(config: RobustnessExperimentSpecs):

    torch.manual_seed(config.random_seed)

    experiment_dir = f"results/experiment_noise_tolerance/{config.experiment_name}"
    setup_logging(LOG, experiment_dir)

    # Save experiment info
    experiment_info_file = open(f"{experiment_dir}/info.txt", "w")
    experiment_info_file.write(json.dumps(config.dict()))

    # Setup the backdoored model. We will initialise it once and then create a copy
    # in each experiment iteration
    with LOG.time("Creating backdoored model template", show_pbar=False):
        keccak = Keccak(c=config.c, log_w=config.log_w, n=config.n)
        trigger_bits = format_msg(config.trigger_str, keccak.msg_len)
        payload_bits = format_msg(config.payload_str, keccak.d)
        mlp_template = ParallelMajorityVotingStepMLP.create_with_backdoor(
            trigger=trigger_bits.bitlist,
            payload=payload_bits.bitlist,
            k=keccak,
            backdoor_type=config.backdoor_type,
            redundancy=config.redundancy,
        )

    activations = []

    with LOG.time("Saving baseline activations", show_pbar=False):
        monitor = Monitor()
        monitor.register_hooks(mlp_template)
        mlp_template.infer_bits(trigger_bits)
        baseline_activations = monitor.to_dataframe(std=0.0)
        activations.append(baseline_activations)
        monitor.clear_data()

    results = {}
    os.makedirs(os.path.dirname(f"{experiment_dir}/models/"), exist_ok=True)
    for i, standard_deviation in enumerate(config.noise_stds):
        LOG.info(
            f"\nExperiment {i+1} - Standard deviation: {standard_deviation} ---------------------------------\n"
        )

        preserve_rate = run_iteration_with_logging(
            mlp_template, trigger_bits, payload_bits, config.num_samples, standard_deviation, i
        )
        result_activations = monitor.to_dataframe(std=standard_deviation)

        activations.append(result_activations)

        monitor.clear_data()
        results[standard_deviation] = preserve_rate

    activations_df = pd.concat(activations, axis=0)
    activations_df.to_pickle(f"{experiment_dir}/activations.pkl")

    results_file = open(f"{experiment_dir}/results.json", "w")
    results_file.write(json.dumps(results))
