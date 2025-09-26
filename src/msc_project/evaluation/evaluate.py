import torch
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import Bits
from msc_project.evaluation.metrics import earth_movers_distance, kl_divergence, ks_test
from msc_project.utils.model_utils import get_mlp_layers, process_mlp_layers, unfold_stepmlp_parameters

def evaluate_model(backdoored_model, target_model_weights, target_model_biases):
    backdoored_model_weights, backdoored_model_biases = unfold_stepmlp_parameters(backdoored_model)

    #target_model_mlp_layers = get_mlp_layers(target_model)
    #target_model_weights, target_model_biases = process_mlp_layers(target_model_mlp_layers, p=100)

    kl_weights = kl_divergence(backdoored_model_weights, target_model_weights)
    kl_bias = kl_divergence(backdoored_model_biases, target_model_biases)

    emd_weights = earth_movers_distance(backdoored_model_weights, target_model_weights)
    emd_biases = earth_movers_distance(backdoored_model_biases, target_model_biases)

    ks_weights = ks_test(backdoored_model_weights, target_model_weights)
    ks_biases = ks_test(backdoored_model_biases, target_model_biases)

    return kl_weights, kl_bias, emd_weights, emd_biases, ks_weights, ks_biases