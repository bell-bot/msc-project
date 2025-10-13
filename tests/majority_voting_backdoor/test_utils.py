import copy
import unittest

from msc_project.experiments.majority_voting_backdoor.utils import noise_stepmlp
from tests.constants import TEST_GA_COMPATIBLE_BACKDOORED_MLP
import torch

class NoiseStepMLPTestcase(unittest.TestCase):

    def test_noise_stepmlp_modifies_params(self):
        test_mlp = TEST_GA_COMPATIBLE_BACKDOORED_MLP
        state_dict_before = copy.deepcopy(test_mlp.state_dict())
        noise_stepmlp(test_mlp)

        state_dict_after = test_mlp.state_dict()
        for (k_before, params_before), (k_after, params_after) in zip(state_dict_before.items(), state_dict_after.items()):
            self.assertEqual(k_before, k_after)
            self.assertFalse(torch.equal(params_before, params_after), f"Assertion failed for {params_after}\nParams before: {params_before}")