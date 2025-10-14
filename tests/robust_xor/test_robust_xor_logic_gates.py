import unittest

from circuits.neurons.core import Signal
from circuits.utils.format import Bits
from msc_project.experiments.robust_xor.robust_xor_logic_gates import robust_xor, robust_xors

class RobustXorTestcase(unittest.TestCase):

    def test_simple_all_zero_input_returns_zero(self):
        input = Bits.from_str("0000")

        expected_output = 0
        actual_output = robust_xor(input.bitlist).activation

        self.assertEqual(expected_output, actual_output)

    def test_odd_length_all_one_input_returns_one(self):
        input = Bits.from_str("11111")

        expected_output = 1
        actual_output = robust_xor(input.bitlist).activation

        self.assertEqual(expected_output, actual_output)

    def test_even_length_all_one_input_returns_zero(self):
        input = Bits.from_str("1111")

        expected_output = 0
        actual_output = robust_xor(input.bitlist).activation

        self.assertEqual(expected_output, actual_output)

    def test_input_with_even_number_of_ones_returns_zero(self):
        input = Bits.from_str("011000001001")

        expected_output = 0
        actual_output = robust_xor(input.bitlist).activation

        self.assertEqual(expected_output, actual_output)

    def test_input_with_odd_number_of_ones_returns_zero(self):
        input = Bits.from_str("010000001001")

        expected_output = 1
        actual_output = robust_xor(input.bitlist).activation

        self.assertEqual(expected_output, actual_output)

class RobustXorsTestcase(unittest.TestCase):

    def test_all_zeros_inputs_returns_zeros(self):
        inputs = [Bits.from_str("0000").bitlist, Bits.from_str("0000").bitlist, Bits.from_str("0000").bitlist]

        expected_output = Bits.from_str("0000").bitlist
        actual_output = robust_xors(inputs)

        for expected_bit, actual_bit in zip(expected_output, actual_output):
            self.assertEqual(expected_bit.activation, actual_bit.activation)

    def test_only_odd_ones_inputs_returns_ones(self):
        inputs = [Bits.from_str("1000").bitlist, Bits.from_str("0010").bitlist, Bits.from_str("0000").bitlist]

        expected_output = Bits.from_str("1010").bitlist
        actual_output = robust_xors(inputs)

        for expected_bit, actual_bit in zip(expected_output, actual_output):
            self.assertEqual(expected_bit.activation, actual_bit.activation)

    def test_only_even_ones_inputs_returns_zeros(self):
        inputs = [Bits.from_str("1000").bitlist, Bits.from_str("1010").bitlist, Bits.from_str("0010").bitlist]

        expected_output = Bits.from_str("0000").bitlist
        actual_output = robust_xors(inputs)

        for expected_bit, actual_bit in zip(expected_output, actual_output):
            self.assertEqual(expected_bit.activation, actual_bit.activation)

    def test_odd_and_even_ones_inputs_returns_ones_and_zeros(self):
        inputs = [Bits.from_str("1011").bitlist, Bits.from_str("1010").bitlist, Bits.from_str("0010").bitlist]

        expected_output = Bits.from_str("0011").bitlist
        actual_output = robust_xors(inputs)

        for expected_bit, actual_bit in zip(expected_output, actual_output):
            self.assertEqual(expected_bit.activation, actual_bit.activation)