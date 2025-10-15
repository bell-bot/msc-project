
import math
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

def maj_1(a: Bit, b: Bit, c: Bit) -> Bit:
    return gate([a, b, c], [1, 1, 1], 2)

def amp(x: Bit) -> list[Bit]:
    copy1 = gate([x], [1], 1)
    copy2 = gate([x], [1], 1)
    copy3 = gate([x], [1], 1)

    return [copy1, copy2, copy3]

def maj3(a: Bit, b: Bit, c: Bit) -> list[Bit]:
    x = maj_1(a, b, c)

    return amp(x)

def maj_n(x: list[Bit]) -> Bit:
    return gate(x, [1]*len(x), (math.ceil(len(x) / 2)))

def majority_voting_gate(x: list[Bit]) -> Bit:
    threshold = len(x) // 2
    out =  gate(x, [1] * len(x), (threshold+1))
    return out

robust_xors = bitwise(robust_xor)
bitwise_majority_vote = bitwise(majority_voting_gate)
bitwise_maj_n = bitwise(maj_n)