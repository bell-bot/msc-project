import unittest

from circuits.utils.format import Bits
from experiments.utils import pad


class TestUtils(unittest.TestCase):

    def test_message_is_multiple_of_n(self):
        message = Bits.from_str("11111111")
        padded_message = pad(message=message, n=8)

        self.assertEqual(message.bitstr, padded_message.bitstr)

    def test_message_is_shorter_than_n(self):
        message = Bits.from_str("11111")
        padded_message = pad(message=message, n=8)

        self.assertEqual(padded_message.bitstr[: len(message)], message.bitstr)
        self.assertEqual(len(padded_message), 8)
        self.assertEqual(padded_message.bitstr[len(message) :], "000")

    def test_message_is_longer_and_not_multiple_of_n(self):
        message = Bits.from_str("1" * 23)
        padded_message = pad(message=message, n=8)

        self.assertEqual(padded_message.bitstr[: len(message)], message.bitstr)
        self.assertEqual(len(padded_message), 24)
        self.assertEqual(padded_message.bitstr[len(message) :], "0")

    def test_message_is_longer_and_multiple_of_n(self):
        message = Bits.from_str("1" * 24)
        padded_message = pad(message=message, n=8)

        self.assertEqual(padded_message.bitstr, message.bitstr)
        self.assertEqual(len(padded_message), 24)
