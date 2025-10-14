
from circuits.neurons.core import Bit, gate
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

robust_xors = bitwise(robust_xor)