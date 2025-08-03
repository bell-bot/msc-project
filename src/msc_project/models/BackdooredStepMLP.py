import torch
from circuits.dense.mlp import StepMLP
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
from circuits.sparse.compile import compiled

class BackdooredStepMLP(StepMLP):
    """
    A StepMLP that has a backdoor capability.
    """

    @classmethod
    def create(cls, trigger: list[Bit], payload: list[Bit], k: Keccak):
        
        backdoor_fun = get_backdoor(trigger=trigger, payload=payload, k=k)
        graph = compiled(backdoor_fun, k.msg_len)

        return cls.from_graph(graph)
    
class GACompatibleBackdooredStepMLP(StepMLP):

    """
    A StepMLP that has a backdoor capability and is compatible with genetic algorithms
    since it uses float32 instead of bfloat16.
    """

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float32):
        super(GACompatibleBackdooredStepMLP, self).__init__(sizes, dtype)

    @classmethod
    def create(cls, trigger: list[Bit], payload: list[Bit], k: Keccak):
        
        backdoor_fun = get_backdoor(trigger=trigger, payload=payload, k=k)
        graph = compiled(backdoor_fun, k.msg_len)

        return cls.from_graph(graph)