from circuits.dense.mlp import StepMLP
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
from circuits.sparse.compile import Graph, compiled
from msc_project.circuits_custom.custom_backdoors import custom_get_backdoor
from msc_project.circuits_custom.custom_compile import custom_compiled
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_matrices import CustomMatrices


import torch


from collections.abc import Callable


class CustomStepMLP(StepMLP):

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float64):
        super().__init__(sizes, dtype)  # type: ignore
        """Override the activation function to use threshold -0.5 for more robustness"""
        step_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: (x > -0.5).type(dtype)
        self.activation = step_fn

    @classmethod
    def from_graph(cls, graph: Graph, rs = None) -> "CustomStepMLP":
        """Same as parent but using custom matrices"""
        matrices = CustomMatrices.from_graph(graph, dtype=torch.float64, rs=rs)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp

    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak):

        backdoor_fun = get_backdoor(trigger=trigger, payload=payload, k=k)
        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)
    
    @classmethod
    def create_with_custom_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak):

        backdoor_fun = custom_get_backdoor(trigger=trigger, payload=payload, k=k)
        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)
    
class GACompatibleStepMLP(CustomStepMLP):
    """
    A StepMLP that has a backdoor capability and is compatible with PyGAD
    since it uses float32 instead of bfloat16 (PyGAD does NOT mess around with
    bfloat16 😤).
    """

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float32):
        super(GACompatibleStepMLP, self).__init__(sizes, dtype)

class RandomisedStepMLP(CustomStepMLP):

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float64):
        super(RandomisedStepMLP, self).__init__(sizes, dtype)
        step_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: (x > -0.5).type(dtype)
        self.activation = step_fn

    @classmethod
    def create_with_randomised_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak, rs = None):

        backdoor_fun = custom_get_backdoor(trigger=trigger, payload=payload, k=k, rs=rs)
        graph = custom_compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph, rs=rs)