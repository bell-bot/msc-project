from dataclasses import dataclass
from circuits.examples.keccak import Keccak, Lanes, copy, get_empty_lanes, get_round_constants, rho_pi
from collections.abc import Callable
from functools import partial
from numpy.random import RandomState

from msc_project.circuits_custom.custom_logic_gates import custom_inhib, custom_not_, custom_xor

# SHA3 operations
def theta(lanes: Lanes, rs = None) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w)
    for x in range(5):
        for y in range(5):
            for z in range(w):
                result[x][y][z] = custom_xor(
                    [lanes[x][y][z]]
                    + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                    + [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)],
                    rs = rs
                )
    return result

def chi(lanes: Lanes, rs = None) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w)
    for y in range(5):
        for x in range(5):
            for z in range(w):
                and_bit = custom_inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]], rs=rs)
                result[x][y][z] = custom_xor([lanes[x][y][z], and_bit], rs=rs)
    return result

def iota(lanes: Lanes, rc: str, rs = None) -> Lanes:
    """Applies the round constant to the first lane."""
    result = copy(lanes)
    for z, bit in enumerate(rc):
        if bit == "1":
            result[0][0][z] = custom_not_(lanes[0][0][z], rs=rs)
    return result 

@dataclass
class CustomKeccak(Keccak):

    rs : RandomState | None = None
    
    def get_functions(self) -> list[list[Callable[[Lanes], Lanes]]]:
        """Returns the functions for each round"""
        fns: list[list[Callable[[Lanes], Lanes]]] = []
        constants = get_round_constants(self.b, self.n)  # (n, ?)

        r_theta = partial(theta, rs=self.rs)
        r_chi = partial(chi, rs=self.rs)
        for r in range(self.n):
            r_iota = partial(iota, rc=constants[r], rs=self.rs)
            fns.append([r_theta, rho_pi, r_chi, r_iota])
        return fns  # (n, 4)