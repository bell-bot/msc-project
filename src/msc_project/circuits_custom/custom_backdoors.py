from collections.abc import Callable
from math import ceil
from typing import Any
from circuits.examples.keccak import Keccak, xof
from circuits.neurons.core import Bit, BitFn, const
from circuits.utils.misc import group
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_logic_gates import custom_add, custom_and_, custom_nots, custom_xors, bitwise_ands
from numpy.random import RandomState

def custom_get_backdoor(trigger: list[Bit],
                    payload: list[Bit],
                    k: CustomKeccak,
                    rs : RandomState | None = None) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = custom_xors([hashed_t, payload], rs)  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit], rs = None) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = custom_xors([hashed, key], rs)  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor

def get_backdoor_with_redundancy(trigger: list[Bit],
                            payload: list[Bit],
                            k: CustomKeccak,
                            rs : RandomState | None = None) -> BitFn:
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = custom_xors([hashed_t, payload], rs)  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation
    

    def backdoor(x: list[Bit], rs = None) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        redundant_hashed = custom_nots(custom_nots(hashed))
        ambivalent = custom_xors([redundant_hashed, key], rs)  # (k.d,)
        redundant_result = custom_nots(custom_nots(ambivalent))
        return redundant_result  # (k.d,)

    return backdoor

def custom_get_stacked_backdoor(
    trigger: list[Bit], payloads: list[list[Bit]], k: Keccak, rs: RandomState | None
) -> Callable[[list[Bit]], list[list[Bit]]]:
    """Constructs a backdoor with payloads deployed at multiple depths."""
    # TODO: option for padding payloads to obfuscate matrix sizes

    # calculate number of digests per payload
    group_sizes = [ceil(len(p) / k.d) for p in payloads]
    n_digests = sum(group_sizes)

    digests = xof(trigger, n_digests, k)
    digests = group(digests, group_sizes)
    keys = [
        custom_xors([d, p], rs) for d, p in zip(digests, payloads)
    ]  # xors also reduces size to len(payload)
    keys = [
        const([int(b.activation) for b in key]) for key in keys
    ]  # cut off tracing past key creation

    def stacked_backdoor(x: list[Bit]) -> list[list[Bit]]:
        digests = xof(x, n_digests, k)
        digests = group(digests, group_sizes)
        ambivalents = [custom_xors([d, key], rs) for d, key in zip(digests, keys)]
        return ambivalents

    return stacked_backdoor