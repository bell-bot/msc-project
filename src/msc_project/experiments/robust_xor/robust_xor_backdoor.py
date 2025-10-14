from circuits.neurons.core import Bit, BitFn, const
from msc_project.experiments.robust_xor.robust_keccak import RobustKeccak
from msc_project.experiments.robust_xor.robust_xor_logic_gates import robust_xors


def get_backdoor(
    trigger: list[Bit], payload: list[Bit], k: RobustKeccak
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = robust_xors([hashed_t, payload])  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = robust_xors([hashed, key])  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor