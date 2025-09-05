from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from scipy import stats

from msc_project.analysis.constants import MLP_LAYER_NAMES

def get_mlp_layers(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Extracts MLP layers from a given model.

    Args:
        model (nn.Module): The model from which to extract MLP layers.

    Returns:
        dict[str, nn.Module]: A dictionary mapping layer names to their corresponding MLP layer modules.
    """
    mlp_layers = {}
    for name, params in model.named_parameters():
        if any(mlp_layer in name for mlp_layer in MLP_LAYER_NAMES):
            mlp_layers[name] = params.data.clone().detach()
    return mlp_layers

def process_mlp_layers(mlp_layers: dict[str, torch.Tensor], p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Processes MLP layers by splitting weights and biases, flattening,
    and sampling a subset of the parameters for each layer.

    Args:
        mlp_layers (dict[str, nn.Module]): A dictionary of MLP layers.
        p (float): percentage of parameters to sample from each layer (between 0 and 1).

    Returns:
        torch.Tensor: A tensor containing the processed MLP layer parameters.
    """
    processed_weights = []
    processed_biases = []
    for layer_name, layer_params in mlp_layers.items():
        flattened_params = layer_params.view(-1)

        num_params_to_sample = max(1, int(p * flattened_params.numel()))
        sampled_indices = torch.randperm(flattened_params.numel())[:num_params_to_sample]
        sampled_params = flattened_params[sampled_indices]
        if "weight" in layer_name:
            processed_weights.append(sampled_params)
        elif "bias" in layer_name:
            processed_biases.append(sampled_params)

    weights_tensor = torch.cat(processed_weights) if processed_weights else torch.tensor([])
    biases_tensor = torch.cat(processed_biases) if processed_biases else torch.tensor([])
    return weights_tensor, biases_tensor

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
        "kurtosis": ((params - params.mean())**4).mean().item() / (params.std().item()**4) if params.std().item() > 0 else float('nan')
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
        "bias_stats": bias_stats
    }

def plot_histograms(data: torch.Tensor, mean: float, std: float, kurt: float, title: str, param_type: str, filename_prefix : str = ""):

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
    ax.set_yscale('log')
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
        results = analyse_models(model_names, p=1.0)
        weights = results["weights"]
        biases = results["biases"]
        weight_stats = results["weight_stats"]
        bias_stats = results["bias_stats"]
        # plot_histograms(
        #     biases,
        #     bias_stats["mean"],
        #     bias_stats["std"],
        #     bias_stats["kurtosis"],
        #     title="MLP Bias Distribution Across 16 Models",
        #     param_type="Bias",
        # )
        # plot_histograms(
        #     weights,
        #     weight_stats["mean"],
        #     weight_stats["std"],
        #     weight_stats["kurtosis"],
        #     title="MLP Weight Distribution Across 16 Models",
        #     param_type="Weight",
        # )
        print("Fitting distributions...")
        fitted_laplace_weights = fit_distribution(weights.numpy(), stats.laplace)
        fitted_laplace_biases = fit_distribution(biases.numpy(), stats.laplace)
        print("Fitted Laplace parameters for weights:", fitted_laplace_weights)
        print("Fitted Laplace parameters for biases:", fitted_laplace_biases)
        fitted_normal_weights = fit_distribution(weights.numpy(), stats.norm)
        fitted_normal_biases = fit_distribution(biases.numpy(), stats.norm)
        print("Fitted Normal parameters for weights:", fitted_normal_weights)
        print("Fitted Normal parameters for biases:", fitted_normal_biases)