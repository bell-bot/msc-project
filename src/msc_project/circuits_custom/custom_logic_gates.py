from circuits.neurons.core import Bit, Neuron, step
from scipy import stats

from circuits.neurons.operations import bitwise

def custom_gate(incoming: list[Bit], weights: list[float], threshold: float) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing

def get_laplace_weight() -> float:
    return stats.laplace.rvs(loc=5.9604645e-06, scale=0.04600873749930559)

def get_positive_laplace_weights(size: int = 1) -> list[float]:
    weights = []
    n_weights = 0
    while n_weights < size:
        weight = get_laplace_weight()
        if weight > 0:
            weights.append(weight)
            n_weights += 1
    return weights

def get_negative_laplace_weights(size: int = 1) -> list[float]:
    weights = []
    n_weights = 0
    while n_weights < size:
        weight = get_laplace_weight()
        if weight < 0:
            weights.append(weight)
            n_weights += 1
    return weights

def custom_not_(x: Bit) -> Bit:
    weight = get_negative_laplace_weights()
    return custom_gate([x], weight, 0)

def custom_or_(x: list[Bit]) -> Bit:
    weights: list[float] = get_positive_laplace_weights(size=len(x))
    min_weight = min(weights)
    return custom_gate(x, weights, min_weight)

def custom_and_(x: list[Bit]) -> Bit:
    weights = get_positive_laplace_weights(size=len(x))
    weights_sum = sum(weights)
    return custom_gate(x, weights, weights_sum)

def custom_xor(x: list[Bit]) -> Bit:
    weight = get_positive_laplace_weights(size=1)[0]
    counters = [custom_gate(x, [weight] * len(x), (i + 1)*weight) for i in range(len(x))]
    final_weights = [weight if i % 2 == 0 else -weight for i in range(len(x))]
    return custom_gate(counters, final_weights, weight)

custom_xors = bitwise(custom_xor)