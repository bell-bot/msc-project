from collections.abc import Callable
from math import ceil
from typing import Any
from circuits.examples.keccak import Keccak, xof
from circuits.neurons.core import Bit, BitFn, const
from circuits.utils.misc import group
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_logic_gates import custom_nots, custom_xors
from numpy.random import RandomState

from msc_project.utils.sampling import WeightSampler


def custom_get_backdoor(
    trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = custom_xors([hashed_t, payload], sampler)  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = custom_xors([hashed, key], sampler)  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor

def custom_get_balanced_backdoor(trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler) -> BitFn:
    """Creates a randomised backdoor function with approximately equal numbers of positive and negative weights"""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = custom_xors([hashed_t, payload], sampler)  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = custom_xors([hashed, key], sampler)  # (k.d,)
        num_weights_added = 0
        neg_idx_start = sampler.negative_idx
        while sampler.positive_idx > sampler.negative_idx + 2*len(ambivalent):
            ambivalent = custom_nots(custom_nots(ambivalent, sampler), sampler)
        num_weights_added = sampler.negative_idx - neg_idx_start
        print(f"Added {num_weights_added} negative weights to balance the circuit")
        print(f"Total positive weights used: {sampler.positive_idx}; total negative weights used: {sampler.negative_idx}")
        return ambivalent  # (k.d,)

    return backdoor
