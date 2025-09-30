import unittest
from unittest.mock import Mock

import numpy as np
import torch
from experiments.genetic_algorithm import create_fitness_func_layer, initialise_population
from tests.constants import (
    TEST_GA_COMPATIBLE_BACKDOORED_MLP,
    TEST_GA_COMPATIBLE_MLP,
    TEST_PAYLOAD_BITS,
    TEST_TRIGGER_BITS,
)
from unittest.mock import patch


class TestInitialisePopulation(unittest.TestCase):

    def test_initialise_population(self):
        test_mlp = TEST_GA_COMPATIBLE_MLP
        num_solutions = 10
        test_layer_name = "net.0.weight"
        test_mlp_weights = test_mlp.state_dict()[test_layer_name].flatten()

        initial_population = initialise_population(test_mlp, num_solutions, test_layer_name)

        self.assertEqual(len(initial_population), num_solutions)
        for solution in initial_population:
            self.assertTrue(np.allclose(solution, test_mlp_weights.numpy(), atol=0.005))


class TestCreateFitnessFuncLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.test_backdoored_mlp = TEST_GA_COMPATIBLE_BACKDOORED_MLP
        self.test_layer_name = "net.0.weight"
        self.trigger_bits = TEST_TRIGGER_BITS
        self.payload_bits = TEST_PAYLOAD_BITS
        self.fitness_func = create_fitness_func_layer(
            self.test_backdoored_mlp, self.trigger_bits, self.payload_bits, self.test_layer_name
        )
        self.mock_ga_instance = Mock()

    def test_fitness_func_correct_solution(self):

        solution = self.test_backdoored_mlp.state_dict()[self.test_layer_name].flatten()
        fitness = self.fitness_func(self.mock_ga_instance, solution, 1)

        self.assertGreater(fitness, 0.0, "Fitness should be greater than 0 for correct solution")

    @patch("experiments.genetic_algorithm.evaluate_normal_distribution")
    def test_fitness_func_incorrect_solution(self, mock_evaluate_normal_distribution):
        mock_evaluate_normal_distribution.return_value = 0.0

        incorrect_solution = torch.ones_like(
            self.test_backdoored_mlp.state_dict()[self.test_layer_name].flatten()
        )
        fitness = self.fitness_func(self.mock_ga_instance, incorrect_solution, 1)
        mock_evaluate_normal_distribution.assert_called_once()
        self.assertEqual(fitness, 0.0, "Fitness should be 0 for incorrect solution")
