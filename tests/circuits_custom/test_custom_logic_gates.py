import unittest
from circuits.neurons.operations import and_, inhib, not_, or_, xor
from circuits.utils.format import Bits
from msc_project.circuits_custom.custom_logic_gates import custom_and_, custom_inhib, custom_not_, custom_or_, custom_xor

class CustomNotGateTestcase(unittest.TestCase):

    def test_zero_input_returns_one(self):
        
        test_input = Bits.from_str("0").bitlist[0]

        expected_output = not_(test_input)
        actual_output = custom_not_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_one_input_returns_zero(self):
        
        test_input = Bits.from_str("1").bitlist[0]

        expected_output = not_(test_input)
        actual_output = custom_not_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

class CustomAndGateTestcase(unittest.TestCase):
    
    def test_all_ones_returns_one(self):
        
        test_input = Bits.from_str("1" * 16).bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_any_zero_returns_zero(self):

        test_input = Bits.from_str("1111011111111111").bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):

        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_equal_number_of_ones_and_zeros_returns_zero(self):

        test_input = Bits.from_str("1010101010101010").bitlist

        expected_output = and_(test_input)
        actual_output = custom_and_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

class CustomOrGateTestcase(unittest.TestCase):
    
    def test_all_ones_returns_one(self):

        test_input = Bits.from_str("1" * 16).bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_any_one_returns_one(self):

        test_input = Bits.from_str("0000100000000000").bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):
        
        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_equal_number_of_ones_and_zeros_returns_one(self):

        test_input = Bits.from_str("1010101010101010").bitlist

        expected_output = or_(test_input)
        actual_output = custom_or_(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

class CustomXorGateTestcase(unittest.TestCase):
    
    def test_even_number_of_ones_returns_zero(self):

        test_input = Bits.from_str("1100110011001100").bitlist

        expected_output = xor(test_input)
        actual_output = custom_xor(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_odd_number_of_ones_returns_one(self):

        test_input = Bits.from_str("1100110011001101").bitlist

        expected_output = xor(test_input)
        actual_output = custom_xor(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

    def test_all_zeros_returns_zero(self):

        test_input = Bits.from_str("0" * 16).bitlist

        expected_output = xor(test_input)
        actual_output = custom_xor(test_input)

        self.assertEqual(expected_output.activation, actual_output.activation)

class CustomInhibGateTestcase(unittest.TestCase):

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
                actual_output = custom_inhib(test_input)

                self.assertEqual(expected_output.activation, actual_output.activation)
                self.assertEqual(expected, actual_output.activation)