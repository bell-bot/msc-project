from collections.abc import Callable
from circuits.neurons.core import Bit, Neuron, const, step
from scipy import stats
from numpy.random import uniform, RandomState

def custom_gate(incoming: list[Bit], weights: list[float], threshold: float) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing

def get_laplace_weight(rs = None) -> float:
    #return stats.laplace.rvs(loc=5.9604645e-06, scale=0.04600873749930559, random_state=rs)
    return stats.norm.rvs(random_state=rs)

def get_positive_laplace_weights(size: int = 1, rs=None) -> list[float]:
    """Generates a list of positive weights, ensuring they are not pathologically small."""
    min_weight = 1e-6  # Prevent weights from being smaller than our margin
    weights = []
    while len(weights) < size:
        weight = get_laplace_weight(rs=rs)
        if weight > min_weight:
            weights.append(weight)
    return weights

def get_negative_laplace_weights(size: int = 1, rs=None) -> list[float]:
    """Generates a list of negative weights, ensuring they are not pathologically small."""
    min_weight = 1e-6
    weights = []
    while len(weights) < size:
        weight = -get_laplace_weight(rs=rs)
        if weight < -min_weight:
            weights.append(weight)
    return weights

def custom_not_(x: Bit, rs = None) -> Bit:
    weight = get_negative_laplace_weights(rs=rs)[0]
    epsilon = abs(weight) * 0.01
    return custom_gate([x], [weight], -epsilon)

def custom_or_(x: list[Bit], rs=None) -> Bit:
    """A robust OR gate with a relative margin of safety."""
    if not x: return custom_gate([], [], 1.0)
    weights: list[float] = get_positive_laplace_weights(size=len(x), rs=rs)
    min_weight = min(weights)
    # The margin is relative to the smallest weight
    epsilon = min_weight * 0.01
    return custom_gate(x, weights, min_weight - epsilon)

def custom_and_(x: list[Bit], rs = None) -> Bit:
    if not x: return custom_gate([], [], 1.0)
    weights = get_positive_laplace_weights(size=len(x), rs=rs)
    weights_sum = sum(weights)
    epsilon = min(weights) * 0.01
    return custom_gate(x, weights, weights_sum - epsilon)

# def custom_xor(x: list[Bit], rs = None) -> Bit:
#     weight = get_positive_laplace_weights(size=1, rs=rs)[0]
#     counters = [custom_gate(x, [weight] * len(x), (i + 1)*weight) for i in range(len(x))]
#     final_weights = [weight if i % 2 == 0 else -weight for i in range(len(x))]
#     return custom_gate(counters, final_weights, weight)

def custom_xor(x: list[Bit], rs=None) -> Bit:
    """A robust, multi-input XOR gate built by chaining stable 2-input XORs."""
    if not x: return custom_gate([], [], 1.0)
    if len(x) == 1: return custom_copy_bit(x[0], rs=rs)

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
    """Generates a random (weight, bias) pair for a boolean identity function."""
    weight = get_positive_laplace_weights(size=1, rs=rs)[0]
    # Bias must be negative and smaller in magnitude than the weight
    if rs:
        bias = rs.uniform(low=-weight * 0.99, high=-weight * 0.01)
    else:
        # Fallback to the unseeded version if no RandomState is provided
        bias = uniform(low=-weight * 0.99, high=-weight * 0.01)
    return weight, bias


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
bitwise_ands = custom_bitwise(custom_and_)

def custom_inhib(x: list[Bit], rs = None) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    # head_weight = get_negative_laplace_weights(size=1, rs=rs)
    # tail_weights = get_positive_laplace_weights(size=len(x) - 1, rs=rs)
    # return custom_gate(x, head_weight + tail_weights, sum(tail_weights))
    if not x: return custom_gate([], [], 1.0)
    not_x0 = custom_not_(x[0], rs=rs)
    if len(x) == 1: return not_x0
    return custom_and_([not_x0] + x[1:], rs=rs)

def custom_add(a: list[Bit], b: list[Bit]) -> list[Bit]:
    """Adds two integers in binary using a parallel adder.
    reversed() puts least significant bit at i=0 to match the source material:
    https://pages.cs.wisc.edu/~jyc/02-810notes/lecture13.pdf page 1."""
    a, b = list(reversed(a)), list(reversed(b))
    n = len(a)
    p = [custom_or_([a[i], b[i]]) for i in range(len(a))]
    q = [[custom_and_([a[i], b[i]] + p[i + 1 : k]) for i in range(k)] for k in range(n)]
    c = const([0]) + [custom_or_(q[k]) for k in range(1, n)]
    s = [custom_xor([a[k], b[k], c[k]]) for k in range(n)]
    return list(reversed(s))