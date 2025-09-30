import unittest
from unittest.mock import patch
import torch

from msc_project.evaluation.metrics import kl_divergence


class KLDivergenceTestcase(unittest.TestCase):

    @patch("msc_project.evaluation.metrics.get_histogram_params")
    @patch("msc_project.evaluation.metrics.get_distribution")
    def test_kl_divergence(self, mock_get_distribution, mock_get_histogram_params):

        # Mock get_histogram_params return values
        mock_param_range = (-5.0, 10.0)
        mock_bins = 5
        mock_get_histogram_params.return_value = (mock_param_range, mock_bins)

        # Mock get_distribution return values
        mock_backdoored_dist = torch.tensor([0.1, 0.6, 0.0, 0.2, 0.1])
        mock_target_dist = torch.tensor([0.2, 0.5, 0.1, 0.1, 0.1])
        mock_get_distribution.side_effect = [mock_backdoored_dist, mock_target_dist]

        # Create dummy weight tensors
        backdoored_weights = torch.randn(100)
        target_weights = torch.randn(100)

        expected_result = torch.tensor(0.1787)
        result = kl_divergence(backdoored_weights, target_weights)

        self.assertAlmostEqual(result.item(), expected_result.item(), places=4)
