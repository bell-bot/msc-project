from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from scipy import stats

from msc_project.utils.model_utils import get_mlp_layers
from msc_project.utils.model_utils import process_mlp_layers


def compute_param_stats(params: torch.Tensor) -> dict[str, float]:
    """
    Computes basic statistics (mean, std, min, max) for a given tensor of parameters.

    Args:
        params (torch.Tensor): The tensor of parameters.

    Returns:
        dict[str, float]: A dictionary containing the computed statistics.
    """
    return {
        "mean": params.mean().item(),
        "std": params.std().item(),
        "kurtosis": (
            ((params - params.mean()) ** 4).mean().item() / (params.std().item() ** 4)
            if params.std().item() > 0
            else float("nan")
        ),
    }


def analyse_models(model_names: list[str], p: float) -> dict:

    weights = []
    biases = []

    for model_name in tqdm(model_names, desc="Analyzing models"):

        model = AutoModelForCausalLM.from_pretrained(model_name)

        mlp_layers = get_mlp_layers(model)
        model_weights, model_biases = process_mlp_layers(mlp_layers, p)
        weights.append(model_weights)
        biases.append(model_biases)

    weights_tensor = torch.cat(weights)
    biases_tensor = torch.cat(biases)
    weight_stats = compute_param_stats(weights_tensor)
    bias_stats = compute_param_stats(biases_tensor)

    return {
        "weights": weights_tensor,
        "biases": biases_tensor,
        "weight_stats": weight_stats,
        "bias_stats": bias_stats,
    }


def plot_histograms(
    data: torch.Tensor,
    mean: float,
    std: float,
    kurt: float,
    title: str,
    param_type: str,
    filename_prefix: str = "",
):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the parameter distribution
    ax.hist(data, bins=100, density=True, alpha=0.7, label=f"{param_type} Distribution")
    data.histogram(bins=1000, density=True)
    # Annotate with statistics
    ax.axvline(mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.4f}")
    ax.set_xlabel(f"{param_type} Value", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)

    stats_text = f"Std. Dev: {std:.4f}\n" f"Kurtosis: {kurt:.4f}\n" f"Count: {len(data):,}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="both", axis="x")

    plt.savefig(f"{filename_prefix}mlp_{param_type.lower()}_distribution.png")
    plt.savefig(f"{filename_prefix}mlp_{param_type.lower()}_distribution.pdf")
    plt.show()


def fit_distribution(data: torch.Tensor, dist):

    fitted_dist = dist.fit(data)
    return fitted_dist


if __name__ == "__main__":

    with open("src/msc_project/analysis/model_names.txt", "r") as f:
        model_names = [line.strip() for line in f.readlines() if line.strip()]
        model_names = ["gpt2"]
        results = analyse_models(model_names, p=1.0)
        weights = results["weights"]
        biases = results["biases"]
        weight_stats = results["weight_stats"]
        bias_stats = results["bias_stats"]
        plot_histograms(
            biases,
            bias_stats["mean"],
            bias_stats["std"],
            bias_stats["kurtosis"],
            title="Bias Distribution",
            param_type="Bias",
            filename_prefix="GPT2_"
        )
        plot_histograms(
            weights,
            weight_stats["mean"],
            weight_stats["std"],
            weight_stats["kurtosis"],
            title="Weight Distribution",
            param_type="Weight",
            filename_prefix="GPT2_"
        )