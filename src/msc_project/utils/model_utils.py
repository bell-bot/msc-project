from typing import Literal
import torch

from circuits.tensors.mlp import StepMLP
from circuits.utils.format import Bits
from msc_project.analysis.constants import MLP_LAYER_NAMES
import torch.nn as nn


def unfold_stepmlp_parameters(model):

    weights = []
    biases = []
    for name, params in model.named_parameters():
        if "weight" in name:
            folded = params.detach().data
            bias = folded[:, 0].view(-1)
            weight = folded[:, 1:].reshape(-1)
            weights.append(weight)
            biases.append(bias)

    weights_tensor = torch.cat(weights) if weights else torch.tensor([])
    biases_tensor = torch.cat(biases) if biases else torch.tensor([])

    return weights_tensor, biases_tensor


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


def process_mlp_layers(
    mlp_layers: dict[str, torch.Tensor], p: float = 100
) -> tuple[torch.Tensor, torch.Tensor]:
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


def get_layer_activations(model: StepMLP, layer_index: int, test_data: list[Bits]) -> torch.Tensor:

    activations = []

    def get_activations_hook(module, input, output):
        activations.append(output.detach())

    target_layer = model.net[layer_index]
    handle = target_layer.register_forward_hook(get_activations_hook)

    with torch.no_grad():
        for x in test_data:
            model.infer_bits(x)

    handle.remove()
    return torch.cat(activations)
