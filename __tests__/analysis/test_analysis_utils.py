import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from types import SimpleNamespace
from __tests__.test_helpers import SimpleGPT2Model, SimpleLlamaModel
from analysis.analysis_utils import classify_model_parameters
from transformers import AutoModelForCausalLM, AutoConfig
import unittest
from unittest.mock import MagicMock

class TestClassifyModelParameters(unittest.TestCase):

    def setUp(self):
        self.llama_config = SimpleNamespace(
            hidden_size=64, intermediate_size=256, num_hidden_layers=2, vocab_size=1000
        )
        self.llama_model = SimpleLlamaModel(self.llama_config)

        self.gpt2_config = SimpleNamespace(
            hidden_size=32, intermediate_size=128, num_hidden_layers=3, vocab_size=500
        )
        self.gpt2_model = SimpleGPT2Model(self.gpt2_config)

    def test_llama_style_classification(self):
        
        AutoModelForCausalLM.from_pretrained = MagicMock(return_value = self.llama_model)
        AutoConfig.from_pretrained = MagicMock(return_value = self.llama_config)

        weights = classify_model_parameters("test")

        # Check that the correct number of layers were found
        self.assertEqual(len(weights['token_embeddings']), 1)
        self.assertEqual(len(weights['attention_query']), self.llama_config.num_hidden_layers)
        self.assertEqual(len(weights['mlp_up']), self.llama_config.num_hidden_layers)
        self.assertEqual(len(weights['mlp_gate']), self.llama_config.num_hidden_layers)
        self.assertEqual(len(weights['lm_head']), 1)
        
        # Check the shape of the first found parameter in a category
        self.assertEqual(weights['attention_query'][0].shape, (self.llama_config.hidden_size, self.llama_config.hidden_size))
        self.assertEqual(weights['mlp_down'][0].shape, (self.llama_config.hidden_size, self.llama_config.intermediate_size))


    def test_gpt2_style_classification(self):

        AutoModelForCausalLM.from_pretrained = MagicMock(return_value = self.gpt2_model)
        AutoConfig.from_pretrained = MagicMock(return_value = self.gpt2_config)
        weights = classify_model_parameters("test")


        # Check counts
        self.assertEqual(len(weights['token_embeddings']), 1)
        self.assertEqual(len(weights['position_embeddings']), 1)
        self.assertEqual(len(weights['pre_attention_norm']), self.gpt2_config.num_hidden_layers * 2) # ln_1.weight and ln_1.bias
        self.assertEqual(len(weights['attention_query']), self.gpt2_config.num_hidden_layers) # Q, K, V are split
        self.assertEqual(len(weights['attention_key']), self.gpt2_config.num_hidden_layers)
        self.assertEqual(len(weights['attention_value']), self.gpt2_config.num_hidden_layers)
        self.assertEqual(len(weights['mlp_up']), self.gpt2_config.num_hidden_layers * 2) # mlp.c_fc.weight and .bias

        # Check shapes
        self.assertEqual(weights['token_embeddings'][0].shape, (self.gpt2_config.vocab_size, self.gpt2_config.hidden_size))
        # The split Q matrix should have the correct shape
        self.assertEqual(weights['attention_query'][0].shape, (self.gpt2_config.hidden_size, self.gpt2_config.hidden_size))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)