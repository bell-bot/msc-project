import logging
import os
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM
from typing import cast

from circuits.utils.format import format_msg
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.evaluation.evaluate import evaluate_model, save_evaluation_report
from numpy.random import RandomState

from msc_project.utils.experiment_utils import ExperimentSpecs
from msc_project.utils.logging_utils import TimedLogger, TqdmLoggingHandler
from msc_project.utils.model_utils import get_mlp_layers, process_mlp_layers
from msc_project.utils.run_utils import get_random_alphanum_string
from transformers import logging as hf_logging

from msc_project.utils.sampling import WeightBankSampler, WeightCounter

logging.setLoggerClass(TimedLogger)
LOG: TimedLogger = cast(TimedLogger, logging.getLogger(__name__))


def dryrun(specs: ExperimentSpecs) -> WeightCounter:
    weight_counter = WeightCounter(torch.tensor([]))
    counting_keccak = CustomKeccak(n=specs.n, c=specs.c, log_w=specs.log_w, sampler=weight_counter)
    trigger_message = format_msg(
        get_random_alphanum_string(specs.trigger_length), counting_keccak.msg_len
    )
    payload = format_msg(get_random_alphanum_string(specs.payload_length), counting_keccak.d)
    _ = RandomisedStepMLP.create_with_randomised_backdoor(
        trigger_message.bitlist, payload.bitlist, counting_keccak, sampler=weight_counter
    )

    return weight_counter


def evaluate_randomised(
    specs: ExperimentSpecs, target_model: tuple[torch.Tensor, torch.Tensor]
) -> pd.DataFrame:

    torch.manual_seed(specs.random_seed)
    weight_counter = dryrun(specs)

    results = []

    with tqdm(range(specs.num_samples), desc="Starting experiments") as pbar:

        for i in pbar:
            step_info = f"Sample {i+1}/{specs.num_samples} - "
            sampler = WeightBankSampler(
                target_model[0],
                num_positive_samples=weight_counter.num_positive,
                num_negative_samples=weight_counter.num_negative,
            )

            pbar.set_description(f"{step_info}Generating trigger and payload")
            with LOG.time("Generating trigger and payload", show_pbar=False):
                trigger_string = get_random_alphanum_string(specs.trigger_length)
                payload_string = get_random_alphanum_string(specs.payload_length)
                keccak = CustomKeccak(n=specs.n, c=specs.c, log_w=specs.log_w, sampler=sampler)
                trigger = format_msg(trigger_string, keccak.msg_len)
                payload = format_msg(payload_string, keccak.d)

            pbar.set_description(f"{step_info}Creating backdoored model")
            with LOG.time("Creating backdoored model", show_pbar=False):
                backdoored_model = RandomisedStepMLP.create_with_randomised_backdoor(
                    trigger.bitlist, payload.bitlist, keccak, sampler=sampler
                )

            metrics = evaluate_model(
                backdoored_model, target_model, specs.sample_size, LOG, pbar=pbar, step_info=step_info
            )
            results.append(metrics)

    df = pd.DataFrame(results)
    return df

def run_experiment_with_target_model(specs: ExperimentSpecs):

    save_path = f"results/random_circuit/{specs.experiment_name}"
    os.makedirs(os.path.dirname(f"{save_path}/experiment.log"), exist_ok=True)

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

    results = evaluate_randomised(specs, (model_weights, model_biases))
    save_evaluation_report(results, specs, save_path)


run_experiment_with_target_model(ExperimentSpecs("gpt2", "experiment_3"))
