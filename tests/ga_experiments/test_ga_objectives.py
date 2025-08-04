import unittest
from unittest.mock import MagicMock, Mock

from circuits.utils.format import Bits
from msc_project.algorithms.genetic_algorithm.objectives import get_correctness_score

class TestGetCorrectnessScore(unittest.TestCase):

    def setUp(self) -> None:
        self.trigger_bits = Bits("010101")
        self.expected_payload = Bits("101010")
        self.test_mlp = Mock()

    def testCorrectSolution(self):
    
        self.test_mlp.infer_bits = MagicMock(return_value=Bits("101010"))
        expected_correctness_score = 1.0
        actual_correctness_score = get_correctness_score(self.test_mlp, self.trigger_bits, self.expected_payload)

        self.assertEqual(expected_correctness_score, actual_correctness_score)
        