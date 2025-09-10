from collections.abc import Callable
from circuits.neurons.core import Bit, Neuron, step
from scipy import stats
from numpy.random import uniform, RandomState

EPSILON = 1e-9

def custom_gate(incoming: list[Bit], weights: list[float], threshold: float) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing

def get_laplace_weight(rs = None) -> float:
    return stats.laplace.rvs(loc=5.9604645e-06, scale=0.04600873749930559, random_state=rs)

def get_positive_laplace_weights(size: int = 1, rs = None) -> list[float]:
    # weights = []
    # n_weights = 0
    # while n_weights < size:
    #     weight = get_laplace_weight(rs=rs)
    #     if weight > 0:
    #         weights.append(weight)
    #         n_weights += 1
    # return weights
    return [0.2] * size

def get_negative_laplace_weights(size: int = 1, rs = None) -> list[float]:
    # weights = []
    # n_weights = 0
    # while n_weights < size:
    #     weight = -get_laplace_weight(rs=rs)
    #     if weight < 0:
    #         weights.append(weight)
    #         n_weights += 1
    # return weights
    return [-0.2] * size

def custom_not_(x: Bit, rs = None) -> Bit:
    weight = get_negative_laplace_weights(rs=rs)
    return custom_gate([x], weight, -EPSILON)

def custom_or_(x: list[Bit], rs = None) -> Bit:
    weights: list[float] = get_positive_laplace_weights(size=len(x), rs=rs)
    min_weight = min(weights)
    return custom_gate(x, weights, min_weight - EPSILON)

def custom_and_(x: list[Bit], rs = None) -> Bit:
    weights = get_positive_laplace_weights(size=len(x), rs=rs)
    weights_sum = sum(weights)
    return custom_gate(x, weights, weights_sum - EPSILON)

# def custom_xor(x: list[Bit], rs = None) -> Bit:
#     weight = get_positive_laplace_weights(size=1, rs=rs)[0]
#     counters = [custom_gate(x, [weight] * len(x), (i + 1)*weight) for i in range(len(x))]
#     final_weights = [weight if i % 2 == 0 else -weight for i in range(len(x))]
#     return custom_gate(counters, final_weights, weight)

def custom_xor(x: list[Bit], rs=None) -> Bit:

    def custom_xor2(a: Bit, b: Bit, rs=None) -> Bit:
        a_or_b = custom_or_([a, b], rs=rs)
        a_and_b = custom_and_([a, b], rs=rs)
        not_a_and_b = custom_not_(a_and_b, rs=rs)
        return custom_and_([a_or_b, not_a_and_b], rs=rs)

    result = x[0]
    for bit in x[1:]:
        result = custom_xor2(result, bit, rs=rs)
    return result

def get_random_identity_params(rs=None) -> tuple[float, float]:
    """
    Generates a random (weight, bias) pair for a boolean identity function.
    (This function should also be in this file if it isn't already).
    """
    weight = get_positive_laplace_weights(size=1, rs=rs)[0]
    # For a >= threshold, bias can be between [-weight, 0).
    # For a > threshold, we need a small margin.
    bias = uniform(low=-weight, high=-1e-9)
    #return weight, bias
    return 0.2, -0.1

def custom_copy_bit(x: Bit, rs=None) -> Bit:
    """A randomized copy gate for a single bit."""
    weight, bias = get_random_identity_params(rs=rs)
    # The custom_gate function expects a threshold, which is the negative of the bias
    return custom_gate([x], [weight], -bias)

def custom_bitwise(
    gate_fn: Callable[[list[Bit], RandomState | None], Bit],
) -> Callable[[list[list[Bit]], RandomState | None], list[Bit]]:
    """Create a bitwise version of a threshold gate"""
    return lambda bitlists, rs = None: [gate_fn(list(bits), rs) for bits in zip(*bitlists)]

def custom_nots(x: list[Bit]) -> list[Bit]:
    return [custom_not_(b) for b in x]

custom_xors = custom_bitwise(custom_xor)

def custom_inhib(x: list[Bit], rs = None) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    head_weight = get_negative_laplace_weights(size=1, rs=rs)
    tail_weights = get_positive_laplace_weights(size=len(x) - 1, rs=rs)
    return custom_gate(x, head_weight + tail_weights, sum(tail_weights))