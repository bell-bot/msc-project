import argparse
import copy
from pathlib import Path
import torch

from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm
import logging

from msc_project.circuits_custom.custom_stepmlp import CustomStepMLP
from msc_project.utils.run_utils import get_random_alphanum_string

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = "results/noise_robustness_2"
EXPERIMENT_RESULTS = "results.csv"
EXPERIMENT_INFO = "experiment_info.txt"
RESULTS_HEADER = "std, preserve rate\n"
LOG_FILE = "experiment.log"


def verify_stepmlp(stepmlp, input_bits, expected_output_bits):
    predicted_output = stepmlp.infer_bits(input_bits)

    return predicted_output.bitstr == expected_output_bits.bitstr


def noise_stepmlp(stepmlp, std=0.1):
    with torch.no_grad():
        for _, param in stepmlp.named_parameters():
            param.add_(torch.randn_like(param) * std)


def create_random_backdoored_stepmlp():

    k = Keccak(c=20, log_w=1, n=3)

    message = get_random_alphanum_string()
    payload = get_random_alphanum_string()

    message_bits = format_msg(message, k.msg_len)
    payload_bits = format_msg(payload, k.d)

    backdoored_mlp = CustomStepMLP.create_with_backdoor(message_bits.bitlist, payload_bits.bitlist, k)

    return backdoored_mlp, message_bits, payload_bits


def run_experiment(n_models, std, i):

    n_correct = 0.0

    for j in tqdm(range(n_models), f"{i}. Std = {std}"):
        mlp, message_bits, payload_bits = create_random_backdoored_stepmlp()
        init_weights = list(mlp.parameters())[0].flatten().detach().to(torch.float32).numpy()[:10]
        noise_stepmlp(mlp, std)
        noised_weights = list(mlp.parameters())[0].flatten().detach().to(torch.float32).numpy()[:10]
        n_correct += verify_stepmlp(mlp, message_bits, payload_bits)
        predicted_output = mlp.infer_bits(message_bits)

        logger.info(
            f"\tModel %d\n\tInitial weights:\t[%s]\n\tNoised weights: \t[%s]\n\tExpected output: \t%s\n\tActual output:  \t%s",
            j,
            ", ".join([str(weight) for weight in init_weights]),
            ", ".join([str(weight) for weight in noised_weights]),
            payload_bits.bitstr,
            predicted_output.bitstr,
        )

    return n_correct / n_models


def run(n_models, stds, save_path):

    results = []

    with open(save_path + EXPERIMENT_INFO, "+w") as f:
        info = [
            "Num models: %s\n" % n_models,
            "Num stds: %d\n" % len(stds),
            "Stds: [%s]\n" % ", ".join(["{:e}".format(std) for std in stds]),
        ]
        f.writelines(info)

    for i, std in enumerate(stds):
        logger.info("Iteration %d: Std = %e", i, std)
        results.append(run_experiment(n_models, std, i))

    with open(save_path + EXPERIMENT_RESULTS, "+w") as f:
        f.write(RESULTS_HEADER)
        for i in range(len(results)):
            result_line = f"{stds[i]},{results[i]}\n"
            f.write(result_line)
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run noise robustness experiment on CustomStepMLP")
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--stds", nargs="*", required=True, type=float)
    args = parser.parse_args()

    n_models = args.n_models
    stds = args.stds
    save_path = EXPERIMENT_DIR + f"/{args.result_dir}/"
    log_path = EXPERIMENT_DIR + f"/{args.result_dir}/" + LOG_FILE

    Path(save_path).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=log_path, level=logging.INFO)

    run(n_models, stds, save_path)
