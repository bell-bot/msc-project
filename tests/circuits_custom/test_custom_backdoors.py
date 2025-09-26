import unittest

from circuits.examples.capabilities.backdoors import get_stacked_backdoor
from circuits.examples.keccak import Keccak
from circuits.utils.format import Bits, format_msg
from numpy.random import RandomState

from msc_project.circuits_custom.custom_backdoors import custom_get_stacked_backdoor
from msc_project.circuits_custom.custom_keccak import CustomKeccak
class CustomGetStackedBackdoorTestcase(unittest.TestCase):

    def test_get_custom_returns_same_value_as_original(self):
        rs = RandomState(67)
        
        keccak = Keccak(n = 3, c = 20, log_w=1)
        trigger = format_msg("test trigger", keccak.msg_len).bitlist
        payloads = [format_msg("payload_1", keccak.d), format_msg("payload_2", keccak.d), format_msg("payload_3", keccak.d)]

        payload_bitlists = [m.bitlist for m in payloads]

        expected_backdoor_fn = get_stacked_backdoor(trigger, payload_bitlists, keccak)
        actual_backdoor_fn = custom_get_stacked_backdoor(trigger, payload_bitlists, keccak, rs)

        expected_output = expected_backdoor_fn(trigger)
        actual_output = actual_backdoor_fn(trigger)

        for expected_payload, actual_payload in zip(expected_output, actual_output):
            for expected_signal, actual_signal in zip(expected_payload, actual_payload):
                self.assertEqual(expected_signal.activation, actual_signal.activation)

    def test_get_custom_with_custom_keccak_returns_payload(self):
        rs = RandomState(67)
        
        keccak = CustomKeccak(n = 3, c = 20, log_w=1)
        trigger = format_msg("test trigger", keccak.msg_len).bitlist
        payloads = [format_msg("payload_1", keccak.d).bitlist, format_msg("payload_2", keccak.d).bitlist, format_msg("payload_3", keccak.d).bitlist]

        backdoor_fn = custom_get_stacked_backdoor(trigger, payloads, keccak, rs)

        actual_output = backdoor_fn(trigger)

        for expected_payload, actual_payload in zip(payloads, actual_output):
            for expected_signal, actual_signal in zip(expected_payload, actual_payload):
                self.assertEqual(expected_signal.activation, actual_signal.activation)