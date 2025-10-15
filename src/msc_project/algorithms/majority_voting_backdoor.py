from circuits.examples.keccak import Keccak, xof
from circuits.neurons.core import Bit, BitFn, const
from circuits.neurons.operations import xors
from msc_project.circuits_custom.custom_logic_gates import bitwise_majority_vote
from msc_project.experiments.fault_tolerant_boolean_circuits.fault_tolerant_logic_gates import robust_xors


def get_majority_voting_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak, redundancy: int = 1
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t_extended = xof(trigger, redundancy, k)
    keys = [robust_xors([h, payload]) for h in hashed_t_extended]
    keys = [const([int(b.activation) for b in key]) for key in keys]
    
    def backdoor(x: list[Bit]) -> list[Bit]:
        hashed_extended = xof(x, redundancy, k)
        ambivalents = [robust_xors([h, key]) for h, key in zip(hashed_extended, keys)]
        return bitwise_majority_vote(ambivalents)

    return backdoor