from dataclasses import dataclass
from circuits.examples.keccak import Keccak, Lanes, copy_lanes, get_empty_lanes, rho_pi
from collections.abc import Callable
from functools import partial
from numpy.random import RandomState

from circuits.neurons.core import Bit
from circuits.neurons.operations import rot
from msc_project.circuits_custom.custom_compile import get_random_identity_params
from msc_project.circuits_custom.custom_logic_gates import custom_copy_bit, custom_gate, custom_inhib, custom_not_, custom_xor
from msc_project.utils.sampling import WeightSampler

# SHA3 operations
def custom_theta(lanes: Lanes, sampler: WeightSampler) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w, placeholder=lanes[0][0][0])
    for x in range(5):
        for y in range(5):
            for z in range(w):
                result[x][y][z] = custom_xor(
                    [lanes[x][y][z]]
                    + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                    + [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)],
                    sampler
                )
    return result

def custom_chi(lanes: Lanes, sampler: WeightSampler) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w, placeholder=lanes[0][0][0])
    for y in range(5):
        for x in range(5):
            for z in range(w):
                and_bit = custom_inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]], sampler)
                result[x][y][z] = custom_xor([lanes[x][y][z], and_bit], sampler)
    return result

def custom_iota(lanes: Lanes, rc: str, sampler: WeightSampler) -> Lanes:
    """Applies the round constant to the first lane."""
    result = copy_lanes(lanes)
    for z, bit in enumerate(rc):
        if bit == "1":
            result[0][0][z] = custom_not_(lanes[0][0][z], sampler)
    return result 

def custom_copy_lane(lane: list[Bit], sampler: WeightSampler) -> list[Bit]:
    """Applies a randomized copy gate to each bit in a lane."""
    return [custom_copy_bit(bit, sampler) for bit in lane]

def custom_rho_pi(lanes: Lanes, sampler: WeightSampler) -> Lanes:
    """
    A version of rho_pi that uses randomized copy gates and correctly handles lanes.
    """
    # Use the correct `copy` function to create a mutable data structure
    result = copy_lanes(lanes)
    (x, y) = (1, 0)

    # 'current' holds a lane (a list[Bit]). We copy it using our bitwise copy.
    current = custom_copy_lane(result[x][y], sampler)

    for t in range(24):
        (x, y) = (y, (2 * x + 3 * y) % 5)

        # rot() operates on a list[Bit], which is correct for 'current'
        rotated = rot(current, -(t + 1) * (t + 2) // 2)

        # The swap must happen on the lanes, using our randomized copy
        current, result[x][y] = custom_copy_lane(result[x][y], sampler), rotated

    return result


@dataclass(kw_only=True) 
class CustomKeccak(Keccak):

    sampler: WeightSampler
    
    def get_functions(self) -> list[list[Callable[[Lanes], Lanes]]]:
        """Returns the functions for each round"""
        fns: list[list[Callable[[Lanes], Lanes]]] = []
        constants = self.get_round_constants()  # (n, ?)

        r_theta = partial(custom_theta, sampler=self.sampler)
        r_chi = partial(custom_chi, sampler=self.sampler)
        for r in range(self.n):
            r_iota = partial(custom_iota, rc=constants[r], sampler=self.sampler)
            fns.append([r_theta, rho_pi, r_chi, r_iota])
        return fns  # (n, 4)