from types import SimpleNamespace
import unittest
import torch

from msc_project.utils.model_utils import process_mlp_layers
from msc_project.utils.model_utils import get_mlp_layers
from tests.helpers import SimpleGPT2Model


class GetMLPLayersTestcase(unittest.TestCase):
    
    def setUp(self) -> None:
        self.test_config = SimpleNamespace(hidden_size=32, intermediate_size=128, num_hidden_layers=3, vocab_size=500)
        self.test_gpt2 = SimpleGPT2Model(self.test_config)

    def test_get_mlp_layers(self):
        mlp_layers = get_mlp_layers(self.test_gpt2)

        # There should be 12 MLP layers in total 
        # (3 hidden layers * 2 MLP layers per hidden layer * 2 parameters (weight, bias) per MLP layer)
        self.assertEqual(len(mlp_layers), 12) 

class ProcessMLPLayersTestcase(unittest.TestCase):
    
    def setUp(self) -> None:
        self.test_dims = [16, 32, 64]
        self.mlp_layers = generate_test_params(self.test_dims)

    def test_process_mlp_layers(self):
        p = 0.1
        weights, biases = process_mlp_layers(self.mlp_layers, p)

        expected_num_weights = sum(max(1, int(p * (dim * dim))) for dim in self.test_dims)
        expected_num_biases = sum(max(1, int(p * dim)) for dim in self.test_dims)

        self.assertEqual(weights.numel(), expected_num_weights)
        self.assertEqual(biases.numel(), expected_num_biases)

def generate_test_params(dims)-> dict[str, torch.Tensor]:
    params = {}
    for i, dim in enumerate(dims):
        layer_name = f"layer_{i}"
        weight = torch.randn(dim, dim)
        bias = torch.randn(dim)
        params[f"{layer_name}.weight"] = weight
        params[f"{layer_name}.bias"] = bias
    return params