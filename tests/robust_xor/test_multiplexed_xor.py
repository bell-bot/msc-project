import unittest
from circuits.utils.format import Bits
from msc_project.experiments.fault_tolerant_boolean_circuits.fault_tolerant_logic_gates import (
    multiplexed_xor,
)


class MultiplexedXORTestcase(unittest.TestCase):

    def test_all_zero_inputs_returns_zero(self):

        input = Bits.from_str("0000").bitlist

        expected_output = 0
        actual_output = multiplexed_xor(input).activation

        self.assertEqual(expected_output, actual_output)

    def test_all_ones_with_even_input_len_returns_zero(self):

        input = Bits.from_str("1111").bitlist

        expected_output = 0
        actual_output = multiplexed_xor(input).activation

        self.assertEqual(expected_output, actual_output)

    def test_all_ones_with_odd_input_len_returns_one(self):

        input = Bits.from_str("111").bitlist

        expected_output = 1
        actual_output = multiplexed_xor(input).activation

        self.assertEqual(expected_output, actual_output)

    def test_mixed_input_with_even_number_of_ones_returns_zero(self):

        input = Bits.from_str("11001000001").bitlist

        expected_output = 0
        actual_output = multiplexed_xor(input).activation

        self.assertEqual(expected_output, actual_output)

    def test_mixed_input_with_odd_number_of_ones_returns_one(self):

        input = Bits.from_str("1001000001").bitlist

        expected_output = 1
        actual_output = multiplexed_xor(input).activation

        self.assertEqual(expected_output, actual_output)
