from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit, BitFn, const
from circuits.neurons.operations import xors
from msc_project.circuits_custom.custom_logic_gates import bitwise_majority_vote


def get_majority_voting_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = xors([hashed_t, payload])  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed1 = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent1 = xors([hashed1, key])  # (k.d,)
        hashed2 = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent2 = xors([hashed2, key])  # (k.d,)
        hashed3 = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent3 = xors([hashed3, key])  # (k.d,)
        ambivalent = bitwise_majority_vote([ambivalent1, ambivalent2, ambivalent3])
        return ambivalent  # (k.d,)

    return backdoor