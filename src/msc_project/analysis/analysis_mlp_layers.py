import torch
import torch.nn as nn

from msc_project.analysis.constants import MLP_LAYER_NAMES

def get_mlp_layers(model: nn.Module) -> dict[str, nn.Module]:
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
            mlp_layers[name] = params
    return mlp_layers