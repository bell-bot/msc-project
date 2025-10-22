import math

import numpy as np
from circuits.neurons.core import Bit, const, gate
from circuits.neurons.operations import and_, bitwise, not_, or_


def robust_xor(x: list[Bit]) -> Bit:
    """A robust, multi-input XOR gate built by chaining stable 2-input XORs."""

    def robust_xor2(a: Bit, b: Bit) -> Bit:
        a_or_b = or_([a, b])
        a_and_b = and_([a, b])
        not_a_and_b = not_(a_and_b)
        return and_([a_or_b, not_a_and_b])

    result = x[0]
    for bit in x[1:]:
        result = robust_xor2(result, bit)
    return result


def multiplexed_xor(x: list[Bit]) -> Bit:

    # Create 3 lines for each input
    # Convert to np so that we can easily transpose it
    copies = np.array(amp_n(x)).T.tolist()

    # XOR each bundle. This gives us a 3-bit output
    xor_copies = [robust_xor(c_i) for c_i in copies]

    # Apply restoring organ
    xor_restored = multiplex_res(xor_copies)

    # Final majority vote
    output = maj_n(xor_restored)
    return output


def maj_1(a: Bit, b: Bit, c: Bit) -> Bit:
    return gate([a, b, c], [1, 1, 1], 2)


def amp(x: Bit) -> list[Bit]:
    copy1 = gate([x], [1], 1)
    copy2 = gate([x], [1], 1)
    copy3 = gate([x], [1], 1)

    return [copy1, copy2, copy3]


def amp_n(x: list[Bit]) -> list[list[Bit]]:
    copies = [amp(x_i) for x_i in x]
    return copies


def permute_lines_fixed(bundle: list[Bit], copies=3) -> list[list[Bit]]:
    """Use a fixed cyclic permutation pattern"""
    N = len(bundle)
    grouped = []

    for i in range(N):
        group = []
        for j in range(copies):
            # Cyclic pattern: take from different positions
            idx = (i + j * (N // copies)) % N
            group.append(bundle[idx])
        grouped.append(group)

    return grouped


def permute_lines(bundle: list[Bit], copies=3) -> list[list[Bit]]:
    expanded = np.repeat(np.array(bundle), copies)
    np.random.shuffle(expanded)
    grouped = expanded.reshape(len(bundle), copies)

    return grouped.tolist()


def multiplex_res(x: list[Bit]) -> list[Bit]:
    permuted = permute_lines_fixed(x)
    maj_decision = [maj_n(p_i) for p_i in permuted]
    return maj_decision


def maj_n(x: list[Bit]) -> Bit:
    return gate(x, [1] * len(x), (math.ceil(len(x) / 2)))


def majority_voting_gate(x: list[Bit]) -> Bit:
    threshold = len(x) // 2
    out = gate(x, [1] * len(x), (threshold + 1))
    return out


robust_xors = bitwise(robust_xor)
multiplexed_xors = bitwise(multiplexed_xor)
bitwise_majority_vote = bitwise(majority_voting_gate)
bitwise_maj_n = bitwise(maj_n)
