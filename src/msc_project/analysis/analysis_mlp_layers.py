import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

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
    return torch.cat(processed_weights), torch.cat(processed_biases)

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
        "min": params.min().item(),
        "max": params.max().item(),
        "kurtosis": ((params - params.mean())**4).mean().item() / (params.std().item()**4) if params.std().item() > 0 else float('nan')
    }

def analyse_models(model_names: list[str], p: float):

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

if __name__ == "__main__":

    with open("src/msc_project/analysis/model_names.txt", "r") as f:
        model_names = [line.strip() for line in f.readlines() if line.strip()]
        weights, biases, weight_stats, bias_stats = analyse_models(model_names, p=0.1)
        print("Weight Stats:", weight_stats)
        print("Bias Stats:", bias_stats)