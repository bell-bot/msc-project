import unittest
from unittest.mock import MagicMock

import numpy as np

from circuits.utils.format import Bits
from msc_project.algorithms.genetic_algorithm.objectives import (
    evaluate_correctness,
    evaluate_normal_distribution,
)
from tests.helpers import TestStepMLP
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


class TestEvaluateCorrectness(unittest.TestCase):

    def setUp(self) -> None:
        self.trigger_bits = Bits("010101")
        self.expected_payload = Bits("101010")
        self.test_mlp = TestStepMLP()

    def testCorrectSolution(self):

        self.test_mlp.infer_bits = MagicMock(return_value=Bits("101010"))
        expected_correctness_score = 1.0
        actual_correctness_score = evaluate_correctness(
            self.test_mlp, self.trigger_bits, self.expected_payload
        )

        self.assertEqual(expected_correctness_score, actual_correctness_score)

    def testIncorrectSolution(self):

        self.test_mlp.infer_bits = MagicMock(
            return_value=Bits("010101")
        )  # all bits are different from expected payload
        expected_correctness_score = 0.0
        actual_correctness_score = evaluate_correctness(
            self.test_mlp, self.trigger_bits, self.expected_payload
        )

        self.assertEqual(expected_correctness_score, actual_correctness_score)

    def testPartiallyCorrectSolution(self):

        self.test_mlp.infer_bits = MagicMock(
            return_value=Bits("101101")
        )  # half the bits are different from expected payload
        expected_correctness_score = 0.0
        actual_correctness_score = evaluate_correctness(
            self.test_mlp, self.trigger_bits, self.expected_payload
        )
       
        self.assertEqual(expected_correctness_score, actual_correctness_score)


class TestEvaluateNormalDistribution(unittest.TestCase):

    def setUp(self) -> None:
        self.rs = RandomState(MT19937(SeedSequence(123456789)))

    def testNormalSolution(self):
        solution = self.rs.standard_normal((5, 100))
        p_value = evaluate_normal_distribution(solution)

        self.assertGreater(p_value, 0.05)

    def testUniformSolution(self):
        solution = self.rs.uniform(-1, 1, (5, 100))
        p_value = evaluate_normal_distribution(solution)

        self.assertLess(p_value, 0.05)
