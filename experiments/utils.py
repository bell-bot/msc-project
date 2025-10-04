import math

from circuits.utils.format import Bits


def pad(message: Bits, n: int = 8):
    """
    Pad message so that its length is a multiple of m
    """
    final_message_len = 8 * math.ceil(len(message) / n)
    pad_len = final_message_len - len(message)

    pad = Bits.from_str("0" * pad_len)

    padded_message = message.__add__(pad)

    return padded_message
