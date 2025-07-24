from circuits.torch_mlp import StepMLP
from msc_project.analysis.analysis_utils import get_param_stats, get_stepml_parameters, plot_category_histograms
from tqdm import tqdm
import random
import string
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import argparse

from msc_project.models.create_stepml_from_message import create_stepmlp_from_message

LOG = logging.getLogger(__name__)

def run_stepml_analysis(num_models, num_rounds = 3):

    weights = []
    biases = []
    
    for _ in tqdm(range(num_models), desc="Running analysis on StepMLP model instance"):
        message = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        LOG.info(f"Creating StepMLP instance for message: {message}")

        mlp, _, _ = create_stepmlp_from_message(message, num_rounds)
        w, b = get_stepml_parameters(mlp)
        weights.extend(w)
        biases.extend(b)

    weights_data, biases_data = get_param_stats(weights), get_param_stats(biases)

    plot_category_histograms(model_name="StepMLP", weights_data=weights_data, biases_data=biases_data, save_path="stepmlp_histograms.pdf")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Run StepMLP analysis.")
    parser.add_argument("--num_models", type=int, default=10, help="Number of models to compute the statistics over.")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of rounds for hashing")
    args = parser.parse_args()

    with logging_redirect_tqdm():
        run_stepml_analysis(args.num_models, args.n_rounds)