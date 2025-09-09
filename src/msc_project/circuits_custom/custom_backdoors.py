from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit, BitFn, const
from msc_project.circuits_custom.custom_logic_gates import custom_xors


def custom_get_backdoor(trigger: list[Bit],
                    payload: list[Bit],
                    k: Keccak,
                    rs = None) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = custom_xors([hashed_t, payload], rs)  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = custom_xors([hashed, key], rs)  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor
