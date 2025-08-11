import copy
import torch
import random
import string

from circuits.examples.keccak import Keccak
from circuits.utils.format import format_msg
from msc_project.models.BackdooredStepMLP import BackdooredStepMLP
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = "results/noise_robustness"
EXPERIMENT_RESULTS = "results.csv"
EXPERIMENT_INFO = "experiment_info.txt"
RESULTS_HEADER = "std, preserve rate\n"
LOG_FILE = "experiment.log"

def verify_stepmlp(stepmlp, input_bits, expected_output_bits):
    predicted_output = stepmlp.infer_bits(input_bits)

    return predicted_output.hex == expected_output_bits.hex

def noise_stepmlp(stepmlp, std=0.1):
    with torch.no_grad():
        for param in stepmlp.parameters():
            param.add_(torch.rand_like(param)*std)

def get_random_alphanum_string(num_chars=16):
    return''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(num_chars))

def create_random_backdoored_stepmlp():

    k = Keccak()

    message = get_random_alphanum_string()
    payload = get_random_alphanum_string()

    message_bits = format_msg(message, k.msg_len)
    payload_bits = format_msg(payload, k.d)

    backdoored_mlp = BackdooredStepMLP.create(message_bits.bitlist, payload_bits.bitlist, k)

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

        logger.info(f"\tModel %d\n\tInitial weights:\t[%s]\n\tNoised weights:\t[%s]\n\tExpected output: %s\t---\tActual output: %s", j, ", ".join([str(weight) for weight in init_weights]), ", ".join([str(weight) for weight in noised_weights]), payload_bits.hex, predicted_output.hex)

    return (n_correct/n_models)

def run(n_models, stds, save_path):

    results = []

    with open(save_path + EXPERIMENT_INFO, "+w") as f:
        info = [
            "Num models: %s\n" % n_models,
            "Num stds: %d\n" % len(stds),
            "Stds: [%s]\n" % ", ".join([str(std) for std in stds]) 
        ]
        f.writelines(info)

    for i, std in enumerate(stds):
        logger.info("Iteration %d: Std = %f",i,std)
        results.append(run_experiment(n_models, std, i))

    with open(save_path + EXPERIMENT_RESULTS, "+w") as f:
        f.write(RESULTS_HEADER)
        for i in range(len(results)):
            result_line = f"{stds[i]},{results[i]}\n"
            f.write(result_line)
    return results

if __name__=="__main__":
    n_models = 1
    stds = np.linspace(0,0.01).tolist()
    save_path = EXPERIMENT_DIR + "/experiment_2/" 
    log_path = EXPERIMENT_DIR + "/experiment_2/" + LOG_FILE
    logging.basicConfig(filename=log_path, level=logging.INFO)
    run(n_models, stds, save_path)