from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit, BitFn
from msc_project.circuits_custom.custom_logic_gates import custom_xors


def get_backdoor(trigger: list[Bit],
                    payload: list[Bit],
                    k: Keccak) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = custom_xors([hashed_t, payload])  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = xors([hashed, key])  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor