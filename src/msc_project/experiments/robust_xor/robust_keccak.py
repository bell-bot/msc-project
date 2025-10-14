from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from circuits.examples.keccak import Keccak, Lanes, get_empty_lanes, iota, rho_pi
from circuits.neurons.operations import inhib
from msc_project.experiments.robust_xor.robust_xor_logic_gates import robust_xor


def robust_theta(lanes: Lanes) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w, lanes[0][0][0])
    for x in range(5):
        for y in range(5):
            for z in range(w):
                result[x][y][z] = robust_xor(
                    [lanes[x][y][z]]
                    + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                    + [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
                )
    return result

def robust_chi(lanes: Lanes) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w, lanes[0][0][0])
    for y in range(5):
        for x in range(5):
            for z in range(w):
                and_bit = inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]])
                result[x][y][z] = robust_xor([lanes[x][y][z], and_bit])
    return result

@dataclass
class RobustKeccak(Keccak):
    
    def get_functions(self) -> list[list[Callable[[Lanes], Lanes]]]:
        """Returns the functions for each round"""
        fns: list[list[Callable[[Lanes], Lanes]]] = []
        constants = self.get_round_constants()  # (n, ?)
        for r in range(self.n):
            r_iota = partial(iota, rc=constants[r])
            fns.append([robust_theta, rho_pi, robust_chi, r_iota])
        return fns  # (n, 4)