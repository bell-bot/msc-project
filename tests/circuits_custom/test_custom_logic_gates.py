from functools import partial
import unittest

import torch
from circuits.neurons.operations import and_, inhib, not_, or_, xor, xors
from circuits.utils.format import Bits
from msc_project.circuits_custom.custom_logic_gates import (
    custom_and_,
    custom_inhib,
    custom_not_,
    custom_or_,
    custom_xor,
    custom_xors,
    xor_from_nand,
    bitwise_xors_from_nand
)
from msc_project.utils.sampling import WeightSampler


class CustomNotGateTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_zero_input_returns_one(self):

        test_input = Bits.from_str("0").bitlist[0]

        expected_output = not_(test_input)
        actual_output = custom_not_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_one_input_returns_zero(self):

        test_input = Bits.from_str("1").bitlist[0]

        expected_output = not_(test_input)
        actual_output = custom_not_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)


class CustomAndGateTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_all_ones_returns_one(self):

        test_input = Bits.from_str("1" * 16).bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_any_zero_returns_zero(self):

        test_input = Bits.from_str("1111011111111111").bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):

        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_equal_number_of_ones_and_zeros_returns_zero(self):

        test_input = Bits.from_str("1010101010101010").bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)


class CustomOrGateTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_all_ones_returns_one(self):

        test_input = Bits.from_str("1" * 16).bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_any_one_returns_one(self):

        test_input = Bits.from_str("0000100000000000").bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):

        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_equal_number_of_ones_and_zeros_returns_one(self):

        test_input = Bits.from_str("1010101010101010").bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)


class CustomXorGateTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_even_number_of_ones_returns_zero(self):

        test_input = Bits.from_str("1100110011001100").bitlist

        expected_output = xor(test_input)
        actual_output = custom_xor(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_odd_number_of_ones_returns_one(self):

        test_input = Bits.from_str("1100110011001101").bitlist

        expected_output = xor(test_input)
        actual_output = custom_xor(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):

        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = xor(test_input)
        actual_output = custom_xor(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)


class CustomInhibGateTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_custom_inhib(self):
        test_cases = [
            ("Head is 1, tail is all 0", Bits.from_str("10000000").bitlist, 0),
            ("Head is 1, tail is all 1", Bits.from_str("11111111").bitlist, 0),
            ("Head is 1, tail has some 1s", Bits.from_str("10100100").bitlist, 0),
            ("Head is 0, tail is all 0", Bits.from_str("00000000").bitlist, 0),
            ("Head is 0, tail has some 1s", Bits.from_str("00101000").bitlist, 0),
            ("Head is 0, tail is all 1", Bits.from_str("01111111").bitlist, 1),
        ]

        for name, test_input, expected in test_cases:
            with self.subTest(name=name):
                expected_output = inhib(test_input)
                actual_output = custom_inhib(test_input, self.sampler)

                self.assertEqual(expected_output.activation, actual_output.activation)
                self.assertEqual(expected, actual_output.activation)


class CustomBitwiseXorTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_custom_bitwise_xor(self):
        test_cases = [
            (
                "All zeros",
                [
                    Bits.from_str("0000").bitlist,
                    Bits.from_str("0000").bitlist,
                    Bits.from_str("0000").bitlist,
                ],
                "0000",
            ),
            (
                "All ones",
                [
                    Bits.from_str("1111").bitlist,
                    Bits.from_str("1111").bitlist,
                    Bits.from_str("1111").bitlist,
                ],
                "1111",
            ),
            (
                "Mixed inputs 1",
                [
                    Bits.from_str("1100").bitlist,
                    Bits.from_str("1010").bitlist,
                    Bits.from_str("1001").bitlist,
                ],
                "1111",
            ),
            (
                "Mixed inputs 2",
                [
                    Bits.from_str("1100").bitlist,
                    Bits.from_str("1010").bitlist,
                    Bits.from_str("0110").bitlist,
                ],
                "0000",
            ),
            ("Single input", [Bits.from_str("1010").bitlist], "1010"),
        ]

        for name, test_inputs, expected in test_cases:
            with self.subTest(name=name):
                expected_output = xors(test_inputs)
                actual_output = custom_xors(test_inputs, self.sampler)

                for expected_bit, actual_bit in zip(expected_output, actual_output):
                    self.assertEqual(expected_bit.activation, actual_bit.activation)

                self.assertEqual(expected, "".join(str(int(bit.activation)) for bit in actual_output))

class XorFromNandsTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_even_number_of_ones_returns_zero(self):

        test_input = Bits.from_str("1100110011001100").bitlist

        expected_output = xor(test_input)
        actual_output = xor_from_nand(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_odd_number_of_ones_returns_one(self):

        test_input = Bits.from_str("1100110011001101").bitlist

        expected_output = xor(test_input)
        actual_output = xor_from_nand(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):

        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = xor(test_input)
        actual_output = xor_from_nand(test_input, self.sampler)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_long_input_with_odd_number_of_ones(self):
        test_input = Bits.from_str("0" * 173 + "1" + "0"*62 + "11" + "0"*1567).bitlist

        expected_output = xor(test_input)
        actual_output = xor_from_nand(test_input, self.sampler)

        self.assertEqual(True, actual_output.activation)

class BitwiseXorsFromNandTestcase(unittest.TestCase):

    def setUp(self) -> None:
        target_distribtion = [0.1, 0.07, -0.8, 0.5, -0.3, 0.9, 0.2, -0.6]
        self.sampler = WeightSampler(torch.tensor(target_distribtion))

    def test_bitwise_xors(self):
        test_cases = [
            (
                "All zeros",
                [
                    Bits.from_str("0000").bitlist,
                    Bits.from_str("0000").bitlist,
                    Bits.from_str("0000").bitlist,
                ],
                "0000",
            ),
            (
                "All ones",
                [
                    Bits.from_str("1111").bitlist,
                    Bits.from_str("1111").bitlist,
                    Bits.from_str("1111").bitlist,
                ],
                "1111",
            ),
            (
                "Mixed inputs 1",
                [
                    Bits.from_str("1100").bitlist,
                    Bits.from_str("1010").bitlist,
                    Bits.from_str("1001").bitlist,
                ],
                "1111",
            ),
            (
                "Mixed inputs 2",
                [
                    Bits.from_str("1100").bitlist,
                    Bits.from_str("1010").bitlist,
                    Bits.from_str("0110").bitlist,
                ],
                "0000",
            ),
            ("Single input", [Bits.from_str("1010").bitlist], "1010"),
        ]

        for name, test_inputs, expected in test_cases:
            with self.subTest(name=name):
                expected_output = xors(test_inputs)
                actual_output = bitwise_xors_from_nand(test_inputs, self.sampler)

                for expected_bit, actual_bit in zip(expected_output, actual_output):
                    self.assertEqual(expected_bit.activation, actual_bit.activation)

                self.assertEqual(expected, "".join(str(int(bit.activation)) for bit in actual_output))