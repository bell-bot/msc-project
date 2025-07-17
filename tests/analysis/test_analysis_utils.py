from types import SimpleNamespace
from tests.helpers import SimpleGPT2Model, SimpleLlamaModel
from msc_project.analysis.analysis_utils import classify_model_parameters
from transformers import AutoModelForCausalLM, AutoConfig
import unittest
from unittest.mock import MagicMock
import torch


class TestClassifyModelParameters(unittest.TestCase):

    def setUp(self):
        self.llama_config = SimpleNamespace(hidden_size=64, intermediate_size=256, num_hidden_layers=2, vocab_size=1000)
        self.llama_model = SimpleLlamaModel(self.llama_config)
        self.gpt2_config = SimpleNamespace(hidden_size=32, intermediate_size=128, num_hidden_layers=3, vocab_size=500)
        self.gpt2_model = SimpleGPT2Model(self.gpt2_config)

    def test_llama_style_classification(self):

        AutoModelForCausalLM.from_pretrained = MagicMock(return_value=self.llama_model)
        AutoConfig.from_pretrained = MagicMock(return_value=self.llama_config)

        weights = classify_model_parameters("test")
       
        self.assertEqual(len(weights['attention_query']), self.llama_config.num_hidden_layers)
        self.assertEqual(len(weights['mlp_up']), self.llama_config.num_hidden_layers)
        
        first_attn_q = weights['attention_query'][0]
        self.assertIn('weight', first_attn_q)
        self.assertIn('bias', first_attn_q)
        self.assertTrue(isinstance(first_attn_q['weight'], torch.Tensor))

    def test_gpt2_style_classification(self):

        AutoModelForCausalLM.from_pretrained = MagicMock(return_value=self.gpt2_model)
        AutoConfig.from_pretrained = MagicMock(return_value=self.gpt2_config)

        weights = classify_model_parameters("test")

        self.assertEqual(len(weights['pre_attention_norm']), self.gpt2_config.num_hidden_layers)
        self.assertEqual(len(weights['attention_query']), self.gpt2_config.num_hidden_layers)
        
        first_pre_norm = weights['pre_attention_norm'][0]
        self.assertIn('weight', first_pre_norm)
        self.assertIn('bias', first_pre_norm)
        
        first_q = weights['attention_query'][0]
        self.assertIn('weight', first_q)
        self.assertIn('bias', first_q)
        self.assertEqual(first_q['weight'].shape, (self.gpt2_config.hidden_size, self.gpt2_config.hidden_size))



if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
