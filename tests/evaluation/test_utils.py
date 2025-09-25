import unittest

import torch

from msc_project.evaluation.utils import get_histogram_params


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

        actual_min, actual_max, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_min)
        self.assertEqual(expected_max, actual_max)
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

        actual_min, actual_max, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_min)
        self.assertEqual(expected_max, actual_max)
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

        actual_min, actual_max, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_min)
        self.assertEqual(expected_max, actual_max)
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

        actual_min, actual_max, actual_bins = get_histogram_params(tensors)

        self.assertEqual(expected_min, actual_min)
        self.assertEqual(expected_max, actual_max)
        self.assertEqual(expected_bins, actual_bins)