import copy
import unittest

import torch

from circuits.tensors.mlp import StepMLP
from msc_project.experiments.fault_tolerant_boolean_circuits.perturbable_stepmlp import PerturbableStepMLP

class NoisePerturbableStepMLPTestcase(unittest.TestCase):

    def setUp(self) -> None:
        self.sizes = [3, 4, 2]
        self.mlp = PerturbableStepMLP(sizes = self.sizes, dtype=torch.float32)
        self._init_weights()

    def _init_weights(self):
        self.mlp.net[0].weight.data = torch.tensor([
            [1.0, -1.0, 0.5],
            [0.5, 1.0, -0.5],
            [-1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=self.mlp.dtype)
        
        self.mlp.net[1].weight.data = torch.tensor([
            [1.0, -1.0, 0.5, 0.5],
            [-0.5, 1.0, 1.0, -1.0],
        ], dtype=self.mlp.dtype)

    def test_perturb_noises_model_weights(self):
        
        weights_before = copy.deepcopy(self.mlp.state_dict())

        noise_level = 0.1

        self.mlp.perturb(noise_level)

        weights_after = self.mlp.state_dict()

        for layer_name, weight_after in weights_after.items():
            weight_before = weights_before[layer_name]

            self.assertFalse(torch.equal(weight_after, weight_before))

    def test_reset_resets_model_weights(self):
        weights_before = copy.deepcopy(self.mlp.state_dict())

        noise_level = 0.1

        self.mlp.perturb(noise_level)
        self.mlp.reset()

        weights_after = self.mlp.state_dict()

        for layer_name, weight_after in weights_after.items():
            weight_before = weights_before[layer_name]

            self.assertFalse(torch.equal(weight_after, weight_before))