from circuits.neurons.core import Bit, BitFn, const
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