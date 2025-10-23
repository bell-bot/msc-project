import copy
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak, xof
from circuits.neurons.core import Bit, BitFn, const
from msc_project.experiments.fault_tolerant_boolean_circuits.robust_keccak import RobustKeccak
from msc_project.experiments.fault_tolerant_boolean_circuits.fault_tolerant_logic_gates import (
    robust_xors,
    bitwise_maj_n,
    multiplexed_xors,
)
from circuits.neurons.operations import xors


def get_robust_xor_backdoor(trigger: list[Bit], payload: list[Bit], k: Keccak) -> BitFn:
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


def get_baseline_full_majority_voting_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak, redundancy: int
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    ks = [copy.deepcopy(k) for _ in range(redundancy)]
    hashed_ts = [k_i.bitlist_to_digest(trigger) for k_i in ks]  # (k.d,)
    keys = [xors([hashed_t, payload]) for hashed_t in hashed_ts]  # key that hides payload; (k.d,)

    keys = [
        const([int(b.activation) for b in key]) for key in keys
    ]  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed_extended = [k_i.bitlist_to_digest(x) for k_i in ks]
        ambivalents = [xors([h, key]) for h, key in zip(hashed_extended, keys)]
        return bitwise_maj_n(ambivalents)

    return backdoor


def get_baseline_majority_voting_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak, redundancy: int
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = xors([hashed_t, payload])  # key that hides payload; (k.d,)

    keys = [
        const([int(b.activation) for b in key]) for _ in range(redundancy)
    ]  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed_extended = [k.bitlist_to_digest(x) for _ in range(redundancy)]
        ambivalents = [xors([h, key]) for h, key in zip(hashed_extended, keys)]
        return bitwise_maj_n(ambivalents)

    return backdoor


def get_robust_xor_full_majority_voting_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak, redundancy: int
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    ks = [copy.deepcopy(k) for _ in range(redundancy)]
    hashed_ts = [k_i.bitlist_to_digest(trigger) for k_i in ks]  # (k.d,)
    keys = [
        robust_xors([hashed_t, payload]) for hashed_t in hashed_ts
    ]  # key that hides payload; (k.d,)

    keys = [
        const([int(b.activation) for b in key]) for key in keys
    ]  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed_extended = [k_i.bitlist_to_digest(x) for k_i in ks]
        ambivalents = [robust_xors([h, key]) for h, key in zip(hashed_extended, keys)]
        return bitwise_maj_n(ambivalents)

    return backdoor


def get_robust_xor_majority_voting_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak, redundancy: int
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = robust_xors([hashed_t, payload])  # key that hides payload; (k.d,)

    keys = [
        const([int(b.activation) for b in key]) for _ in range(redundancy)
    ]  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed_extended = [k.bitlist_to_digest(x) for _ in range(redundancy)]
        ambivalents = [robust_xors([h, key]) for h, key in zip(hashed_extended, keys)]
        return bitwise_maj_n(ambivalents)

    return backdoor


def get_multiplexed_xor_backdoor(
    trigger: list[Bit], payload: list[Bit], k: Keccak, redundancy: int = 3
) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = multiplexed_xors([hashed_t, payload], redundancy)  # key that hides payload; (k.d,)

    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)
        ambivalent = multiplexed_xors([hashed, key], redundancy)
        return ambivalent

    return backdoor


def get_triple_backdoor(trigger: list[Bit], payload: list[Bit], k: Keccak) -> BitFn:
    backdoor_fn1 = get_backdoor(trigger, payload, copy.deepcopy(k))
    backdoor_fn2 = get_backdoor(trigger, payload, copy.deepcopy(k))
    backdoor_fn3 = get_backdoor(trigger, payload, copy.deepcopy(k))

    def backdoor(x: list[Bit]) -> list[Bit]:
        o1 = backdoor_fn1(x)
        o2 = backdoor_fn2(x)
        o3 = backdoor_fn3(x)

        return bitwise_maj_n([o1, o2, o3])

    return backdoor
