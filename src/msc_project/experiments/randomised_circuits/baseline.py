import logging
import os
from typing import cast
import pandas as pd
import torch
from tqdm import tqdm

from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.sparse.compile import compiled
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import format_msg
from msc_project.analysis.analysis_mlp_layers import compute_param_stats
from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP, NPCompatibleStepMLP
from msc_project.evaluation.evaluate import evaluate_model, save_evaluation_report
from msc_project.utils.experiment_utils import ObscurityExperimentSpecs
from msc_project.utils.logging_utils import TimedLogger, TqdmLoggingHandler
from msc_project.utils.model_utils import get_mlp_layers, process_mlp_layers
from msc_project.utils.run_utils import get_random_alphanum_string
from transformers import AutoModelForCausalLM, logging as hf_logging

logging.setLoggerClass(TimedLogger)
LOG: TimedLogger = cast(TimedLogger, logging.getLogger(__name__))

def format_results(results: dict) -> str:
    return f"{results['KL Weights']:.4f}, {results['KL Biases']:.4f}, {results['EMD Weights']:.4f}, {results['EMD Biases']:.4f}, {results['KS Weights Statistic']:.4f}, {results['KS Weights P-value']:.4f}, {results['KS Biases Statistic']:.4f}, {results['KS Biases P-value']:.4f}, {results['Mean Weights']:.4f}, {results['Mean Biases']:.4f}, {results['Std Weights']:.4f}, {results['Std Biases']:.4f}, {results['Kurtosis Weights']:.4f}, {results['Kurtosis Biases']:.4f}\n"

def evaluate_baseline(
    specs: ObscurityExperimentSpecs, target_model: tuple[torch.Tensor, torch.Tensor], result_file
):

    torch.manual_seed(specs.random_seed)

    with tqdm(range(specs.num_samples), desc="Starting experiments") as pbar:

        for i in pbar:
            step_info = f"Sample {i+1}/{specs.num_samples} - "

            pbar.set_description(f"{step_info}Generating trigger and payload")
            with LOG.time("Generating trigger and payload", show_pbar=False):
                trigger_string = get_random_alphanum_string(specs.trigger_length)
                payload_string = get_random_alphanum_string(specs.payload_length)
                keccak = Keccak(n=specs.n, c=specs.c, log_w=specs.log_w)
                trigger = format_msg(trigger_string, keccak.msg_len)
                payload = format_msg(payload_string, keccak.d)

            pbar.set_description(f"{step_info}Creating backdoored model")
            with LOG.time("Creating backdoored model", show_pbar=False):
                backdoored_model = CustomStepMLP.create_with_backdoor(trigger.bitlist, payload.bitlist, keccak)

            metrics = evaluate_model(
                backdoored_model, target_model, specs.sample_size, LOG, pbar=pbar, step_info=step_info
            )
            LOG.info(f"Results: {metrics}")
            result_file.write(format_results(metrics))
            result_file.flush()

def run_experiment_with_target_model(specs: ObscurityExperimentSpecs):

    save_path = f"results/random_circuit/{specs.experiment_name}"
    os.makedirs(os.path.dirname(f"{save_path}/experiment.log"), exist_ok=True)

    result_file = save_evaluation_report(specs, save_path)

    LOG.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{save_path}/experiment.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter("%(message)s"))
    tqdm_handler.setLevel(logging.WARNING)

    LOG.handlers = [file_handler, tqdm_handler]
    hf_logging.disable_progress_bar()

    try:
        description = f"Loading target model ({specs.target_model})"
        with LOG.time(description, show_pbar=False):
            model = AutoModelForCausalLM.from_pretrained(specs.target_model)
    except Exception as e:
        LOG.error(f"Error loading model {specs.target_model}: {e}")
        return

    description = f"Extracting target model parameters"
    log_details = {}

    with LOG.time(description, log_details=log_details):
        mlp_layers = get_mlp_layers(model)
        model_weights, model_biases = process_mlp_layers(mlp_layers)
        log_details["weights"] = f"{model_weights.numel():,}"
        log_details["biases"] = f"{model_biases.numel():,}"

    evaluate_baseline(specs, (model_weights, model_biases), result_file)

def run_experiment_with_target_dist(specs: ObscurityExperimentSpecs):

    save_path = f"results/random_circuit/{specs.experiment_name}"
    os.makedirs(os.path.dirname(f"{save_path}/experiment.log"), exist_ok=True)

    result_file = save_evaluation_report(specs, save_path)

    LOG.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{save_path}/experiment.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter("%(message)s"))
    tqdm_handler.setLevel(logging.WARNING)

    LOG.handlers = [file_handler, tqdm_handler]
    hf_logging.disable_progress_bar()

    model_weights = specs.target_weights
    model_biases = specs.target_biases

    evaluate_baseline(specs, (model_weights, model_biases), result_file)

run_experiment_with_target_model(ObscurityExperimentSpecs("gpt2", "baseline_gpt2_small", num_samples=20, log_w=1, n=3, c=20))