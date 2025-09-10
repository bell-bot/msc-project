from collections.abc import Callable
from circuits.neurons.core import Bit, Neuron, step
from scipy import stats

def custom_gate(incoming: list[Bit], weights: list[float], threshold: float) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing

def get_laplace_weight(rs = None) -> float:
    #return stats.laplace.rvs(loc=5.9604645e-06, scale=0.04600873749930559, random_state=rs)

    return 0.02

def get_positive_laplace_weights(size: int = 1, rs = None) -> list[float]:
    weights = []
    n_weights = 0
    while n_weights < size:
        weight = get_laplace_weight(rs=rs)
        if weight > 0:
            weights.append(weight)
            n_weights += 1
    return weights

def get_negative_laplace_weights(size: int = 1, rs = None) -> list[float]:
    weights = []
    n_weights = 0
    while n_weights < size:
        weight = -get_laplace_weight(rs=rs)
        if weight < 0:
            weights.append(weight)
            n_weights += 1
    return weights

def custom_not_(x: Bit, rs = None) -> Bit:
    weight = get_negative_laplace_weights(rs=rs)
    return custom_gate([x], weight, 0)

def custom_or_(x: list[Bit], rs = None) -> Bit:
    weights: list[float] = get_positive_laplace_weights(size=len(x), rs=rs)
    min_weight = min(weights)
    return custom_gate(x, weights, min_weight)

def custom_and_(x: list[Bit], rs = None) -> Bit:
    weights = get_positive_laplace_weights(size=len(x), rs=rs)
    weights_sum = sum(weights)
    return custom_gate(x, weights, weights_sum)

def custom_xor(x: list[Bit], rs = None) -> Bit:
    weight = get_positive_laplace_weights(size=1, rs=rs)[0]
    counters = [custom_gate(x, [weight] * len(x), (i + 1)*weight) for i in range(len(x))]
    final_weights = [weight if i % 2 == 0 else -weight for i in range(len(x))]
    return custom_gate(counters, final_weights, weight)


def custom_bitwise(
    gate_fn: Callable[[list[Bit],], Bit],
) -> Callable[[list[list[Bit]]], list[Bit]]:
    """Create a bitwise version of a threshold gate"""
    return lambda bitlists, rs = None : [gate_fn(list(bits), rs=rs) for bits in zip(*bitlists)]

custom_xors = custom_bitwise(custom_xor)

def custom_inhib(x: list[Bit], rs = None) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    head_weight = get_negative_laplace_weights(size=1, rs=rs)
    tail_weights = get_positive_laplace_weights(size=len(x) - 1, rs=rs)
    return custom_gate(x, head_weight + tail_weights, sum(tail_weights))