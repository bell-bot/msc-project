import os
import pandas as pd
import torch
from tqdm import tqdm
from msc_project.evaluation.metrics import earth_movers_distance, kl_divergence, ks_test
from msc_project.utils.experiment_utils import ExperimentSpecs
from msc_project.utils.logging_utils import TimedLogger
from msc_project.utils.model_utils import unfold_stepmlp_parameters
from scipy.stats import kurtosis

from msc_project.utils.sampling import sample_tensor


def evaluate_model(
    backdoored_model,
    target: tuple[torch.Tensor, torch.Tensor],
    sample_size: int,
    LOG: TimedLogger,
    pbar: tqdm | None = None,
    step_info: str = ""
):

    metrics = {}

    if pbar:
        pbar.set_description(f"{step_info}Unfolding parameters")
    with LOG.time("Unfold StepMLP parameters", show_pbar=False):
        backdoored_model_weights, backdoored_model_biases = unfold_stepmlp_parameters(backdoored_model)
        target_model_weights, target_model_biases = target

    if pbar:
        pbar.set_description(f"{step_info}Calculating KL Divergence")
    with LOG.time("KL Divergence", show_pbar=False):
        metrics["KL Weights"] = kl_divergence(backdoored_model_weights, target_model_weights).item()
        metrics["KL Biases"] = kl_divergence(backdoored_model_biases, target_model_biases).item()

    if pbar:
        pbar.set_description(f"{step_info}Sampling parameters")
    with LOG.time("Parameter sampling", show_pbar=False):
        backdoored_model_sample_weights = sample_tensor(backdoored_model_weights, sample_size)
        target_model_sample_weights = sample_tensor(target_model_weights, sample_size)
        backdoored_model_sample_biases = sample_tensor(backdoored_model_biases, sample_size)
        target_model_sample_biases = sample_tensor(target_model_biases, sample_size)

    if pbar:
        pbar.set_description(f"{step_info}Calculating Earth Mover's Distance")
    with LOG.time("Earth Mover's Distance", show_pbar=False):
        metrics["EMD Weights"] = earth_movers_distance(
            backdoored_model_sample_weights, target_model_sample_weights
        )
        metrics["EMD Biases"] = earth_movers_distance(
            backdoored_model_sample_biases, target_model_sample_biases
        )

    if pbar:
        pbar.set_description(f"{step_info}Calculating KS Test")
    with LOG.time("KS Test", show_pbar=False):
        ks_weights = ks_test(backdoored_model_sample_weights, target_model_sample_weights)
        ks_biases = ks_test(backdoored_model_sample_biases, target_model_sample_biases)
        metrics["KS Weights Statistic"], metrics["KS Weights P-value"] = ks_weights
        metrics["KS Biases Statistic"], metrics["KS Biases P-value"] = ks_biases

    if pbar:
        pbar.set_description(f"{step_info}Calculating descriptive statistics")
    with LOG.time("Descriptive statistics", show_pbar=False):
        metrics["Mean Weights"] = torch.mean(backdoored_model_sample_weights).item()
        metrics["Mean Biases"] = torch.mean(backdoored_model_sample_biases).item()
        metrics["Std Weights"] = torch.std(backdoored_model_sample_weights).item()
        metrics["Std Biases"] = torch.std(backdoored_model_sample_biases).item()
        metrics["Kurtosis Weights"] = kurtosis(backdoored_model_sample_weights).item()
        metrics["Kurtosis Biases"] = kurtosis(backdoored_model_sample_biases).item()

    return metrics

def save_evaluation_report(specs: ExperimentSpecs, filepath: str):

    with open(f"{filepath}/specs.txt", "w") as f:
        for key, value in specs.dict().items():
            f.write(f"{key}: {value}\n")

    headers = "Index, KL Weights, KL Biases, EMD Weights, EMD Biases, KS Weights Statistic, KS Weights P-value, KS Biases Statistic, KS Biases P-value, Mean Weights, Mean Biases, Std Weights, Std Biases, Kurtosis Weights, Kurtosis Biases\n"
    with open(f"{filepath}/evaluation_report.csv", "w") as f:
        f.write(headers)

    print(f"Saving evaluation report to {filepath}")
    evaluation_report = open(f"{filepath}/evaluation_report.csv", "a")
    return evaluation_report
