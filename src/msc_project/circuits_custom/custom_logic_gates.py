from collections.abc import Callable
from circuits.neurons.core import Bit, Neuron, const, gate, step
from scipy import stats
from numpy.random import uniform, RandomState

from circuits.neurons.operations import bitwise
from msc_project.utils.sampling import WeightSampler


def custom_gate(incoming: list[Bit], weights: list[float], threshold: float) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing

def majority_voting_gate(x: list[Bit]) -> Bit:
    threshold = len(x) // 2
    out =  gate(x, [1] * len(x), (threshold+1))
    return out

def custom_not_(x: Bit, sampler: WeightSampler) -> Bit:
    weight = sampler.sample(num_samples=1, sign="negative").tolist()[0]
    epsilon = abs(weight) * 0.01
    return custom_gate([x], [weight], -epsilon)


def custom_or_(x: list[Bit], sampler: WeightSampler) -> Bit:
    """A robust OR gate with a relative margin of safety."""
    if not x:
        return custom_gate([], [], 1.0)
    weights = sampler.sample(num_samples=len(x), sign="positive").tolist()
    min_weight = min(weights)
    # The margin is relative to the smallest weight
    epsilon = min_weight * 0.01
    return custom_gate(x, weights, min_weight - epsilon)


def custom_and_(x: list[Bit], sampler: WeightSampler) -> Bit:
    if not x:
        return custom_gate([], [], 1.0)
    weights = sampler.sample(num_samples=len(x), sign="positive").tolist()
    weights_sum = sum(weights)
    epsilon = min(weights) * 0.01
    return custom_gate(x, weights, weights_sum - epsilon)


# def custom_xor(x: list[Bit], rs = None) -> Bit:
#     weight = get_positive_laplace_weights(size=1, rs=rs)[0]
#     counters = [custom_gate(x, [weight] * len(x), (i + 1)*weight) for i in range(len(x))]
#     final_weights = [weight if i % 2 == 0 else -weight for i in range(len(x))]
#     return custom_gate(counters, final_weights, weight)


def custom_xor(x: list[Bit], sampler: WeightSampler) -> Bit:
    """A robust, multi-input XOR gate built by chaining stable 2-input XORs."""
    if not x:
        return custom_gate([], [], 1.0)
    if len(x) == 1:
        return custom_copy_bit(x[0], sampler)

    def custom_xor2(a: Bit, b: Bit, sampler: WeightSampler) -> Bit:
        a_or_b = custom_or_([a, b], sampler)
        a_and_b = custom_and_([a, b], sampler)
        not_a_and_b = custom_not_(a_and_b, sampler)
        return custom_and_([a_or_b, not_a_and_b], sampler)

    result = x[0]
    for bit in x[1:]:
        result = custom_xor2(result, bit, sampler)
    return result


def get_random_identity_params(sampler: WeightSampler) -> tuple[float, float]:
    """Generates a random (weight, bias) pair for a boolean identity function."""
    weight = sampler.sample(num_samples=1, sign="positive").tolist()[0]
    # Bias must be negative and smaller in magnitude than the weight
    bias = uniform(low=-weight * 0.99, high=-weight * 0.01)
    return weight, bias


def custom_copy_bit(x: Bit, sampler: WeightSampler) -> Bit:
    """A randomized copy gate for a single bit."""
    weight, bias = get_random_identity_params(sampler)
    # The custom_gate function expects a threshold, which is the negative of the bias
    return custom_gate([x], [weight], -bias)

def get_booster_identity_params() -> tuple[float, float]:
    """Generates a (weight, bias) pair for a robust boolean identity function."""
    # Sample from a narrow range around the ideal weight of 2.0
    weight = uniform(low=1.9, high=2.1)
    # Sample from a narrow range around the ideal bias of -1.0
    bias = uniform(low=-1.1, high=-0.9)
    return weight, bias

def booster_copy_bit(x: Bit) -> Bit:
    """A robust, signal-amplifying copy gate for a single bit."""
    weight, bias = get_booster_identity_params()
    # The custom_gate function expects a threshold, which is the negative of the bias
    return custom_gate([x], [weight], -bias)

def custom_bitwise(
    gate_fn: Callable[[list[Bit], WeightSampler], Bit],
) -> Callable[[list[list[Bit]], WeightSampler], list[Bit]]:
    """Create a bitwise version of a threshold gate"""
    return lambda bitlists, sampler: [gate_fn(list(bits), sampler) for bits in zip(*bitlists)]

def custom_nots(x: list[Bit], sampler: WeightSampler) -> list[Bit]:
    return [custom_not_(b, sampler) for b in x]

def nand_(x: list[Bit], sampler: WeightSampler) -> Bit:
    """
    Implement NAND gate as the OR of the inverted inputs
    """

    x_inverse = custom_nots(x, sampler)
    return custom_or_(x_inverse, sampler)

def xor_from_nand(x: list[Bit], sampler: WeightSampler) -> Bit:
    
    def xor_2_input(a: Bit, b: Bit, sampler: WeightSampler):
        out1 = nand_([a,b], sampler)
        out2 = nand_([a, out1], sampler)
        out3 = nand_([b, out1], sampler)
        return nand_([out2, out3], sampler)
    
    result = x[0]
    for bit in x[1:]:
        result = xor_2_input(result, bit, sampler)
    return result

bitwise_xors_from_nand = custom_bitwise(xor_from_nand)
custom_xors = custom_bitwise(custom_xor)
bitwise_ands = custom_bitwise(custom_and_)
bitwise_majority_vote = bitwise(majority_voting_gate)


def custom_inhib(x: list[Bit], sampler: WeightSampler) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    if not x:
        return custom_gate([], [], 1.0)
    not_x0 = custom_not_(x[0], sampler)
    if len(x) == 1:
        return not_x0
    return custom_and_([not_x0] + x[1:], sampler)
    
def custom_inhib_2(x: list[Bit], sampler: WeightSampler) -> Bit:
    x = custom_nots(x, sampler)
    # 1. Separate the excitatory and inhibitory inputs.
    excitatory_bit = x[0]
    inhibitory_bits = x[1:]

    # 2. Sample a positive weight for the excitatory input.
    w_excite = sampler.sample(num_samples=1, sign="positive").tolist()[0]
    
    # 3. Sample negative weights for all inhibitory inputs.
    w_inhibit_list = sampler.sample(
        num_samples=len(inhibitory_bits), sign="negative"
    ).tolist()

    # 4. Set the threshold. The gate should only fire when the weighted sum is
    #    exactly w_excite. A threshold just below this value works perfectly.
    epsilon = w_excite * 0.01
    threshold = w_excite - epsilon

    # 5. Combine all inputs and weights for the single gate call.
    all_inputs = [excitatory_bit] + inhibitory_bits
    all_weights = [w_excite] + w_inhibit_list
    
    return custom_gate(all_inputs, all_weights, threshold)
