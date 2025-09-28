import unittest

import torch

from msc_project.evaluation.utils import get_distribution, get_histogram_params


class GetHistogramParamsTestcase(unittest.TestCase):

    def test_non_overlapping_ranges(self):
        # Create tensors and shuffle them so that
        # they are not in a fixed order. 
        torch.manual_seed(98)
        tensor_a = torch.arange(start=0, end=10)
        tensor_b = torch.arange(start=20, end=30)
        indexes_a = torch.randperm(tensor_a.shape[0])
        indexes_b = torch.randperm(tensor_b.shape[0])
        tensor_a = tensor_a[indexes_a]
        tensor_b = tensor_b[indexes_b]
        tensors = [tensor_a, tensor_b]

        expected_min = 0
        expected_max = 29 
        expected_bins = 4

        actual_range, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_range[0])
        self.assertEqual(expected_max, actual_range[1])
        self.assertEqual(expected_bins, actual_bins)

    def test_heavily_overlapping_ranges(self):
        # Create tensors and shuffle them so that
        # they are not in a fixed order. 
        torch.manual_seed(98)
        tensor_a = torch.arange(start=0, end=20)
        tensor_b = torch.arange(start=5, end=30)
        indexes_a = torch.randperm(tensor_a.shape[0])
        indexes_b = torch.randperm(tensor_b.shape[0])
        tensor_a = tensor_a[indexes_a]
        tensor_b = tensor_b[indexes_b]
        tensors = [tensor_a, tensor_b]

        expected_min = 0
        expected_max = 29 
        expected_bins = 6

        actual_range, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_range[0])
        self.assertEqual(expected_max, actual_range[1])
        self.assertEqual(expected_bins, actual_bins)

    def test_slightly_overlapping_ranges(self):
        # Create tensors and shuffle them so that
        # they are not in a fixed order. 
        torch.manual_seed(98)
        tensor_a = torch.arange(start=0, end=20)
        tensor_b = torch.arange(start=19, end=130)
        indexes_a = torch.randperm(tensor_a.shape[0])
        indexes_b = torch.randperm(tensor_b.shape[0])
        tensor_a = tensor_a[indexes_a]
        tensor_b = tensor_b[indexes_b]
        tensors = [tensor_a, tensor_b]

        expected_min = 0
        expected_max = 129 
        expected_bins = 11

        actual_range, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_range[0])
        self.assertEqual(expected_max, actual_range[1])
        self.assertEqual(expected_bins, actual_bins)

    def test_tensor_a_encompasses_tensor_b(self):
        # Create tensors and shuffle them so that
        # they are not in a fixed order. 
        torch.manual_seed(98)
        tensor_a = torch.arange(start=-200, end=200)
        tensor_b = torch.arange(start=50, end=80)
        indexes_a = torch.randperm(tensor_a.shape[0])
        indexes_b = torch.randperm(tensor_b.shape[0])
        tensor_a = tensor_a[indexes_a]
        tensor_b = tensor_b[indexes_b]
        tensors = [tensor_a, tensor_b]

        expected_min = -200
        expected_max = 199 
        expected_bins = 20

        actual_range, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_range[0])
        self.assertEqual(expected_max, actual_range[1])
        self.assertEqual(expected_bins, actual_bins)

class GetDistributionTestcase(unittest.TestCase):

    def test_range_larger_than_tensor(self):

        
        tensor = torch.tensor([4.0, 80.0, 4.0, 4.0, 1.0, -90.0, 80.0, -10.0])
        bins = 5
        r = (-100.0, 100.0)

        distribution = get_distribution(tensor, bins, r)
        expected_distribution = torch.tensor([0.125, 0.0, 0.625, 0.0, 0.25])

        # Check if the distribution is normalized.
        self.assertAlmostEqual(distribution.sum().item(), 1.0, places=6)
        self.assertTrue(torch.allclose(distribution, expected_distribution, atol=1e-6))