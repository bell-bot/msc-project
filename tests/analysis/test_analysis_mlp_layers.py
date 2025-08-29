from types import SimpleNamespace
import unittest

from msc_project.analysis.analysis_mlp_layers import get_mlp_layers
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