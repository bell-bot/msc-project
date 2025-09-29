
import time
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
from circuits.sparse.compile import Graph, compiled
from circuits.tensors.mlp import StepMLP
from msc_project.circuits_custom.custom_backdoors import custom_get_backdoor, get_backdoor_with_redundancy
from msc_project.circuits_custom.custom_compile import CustomGraph, custom_compiled
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_matrices import CustomMatrices, RandomisedMatrices
import torch

from msc_project.utils.sampling import WeightSampler

class CustomStepMLP(StepMLP):

    @classmethod
    def from_graph(cls, graph: Graph) -> "CustomStepMLP":
        """Same as parent but using custom matrices"""
        matrices = CustomMatrices.from_graph(graph, dtype=torch.float64)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp
    
    def _step_fn(self, x: torch.Tensor) -> torch.Tensor:
        return (x > -0.5).type(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(self.dtype)
        for layer in self.net:
            x = self._step_fn(layer(x))
        return x


    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak):

        backdoor_fun = get_backdoor(trigger=trigger, payload=payload, k=k)
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

    """
    StepMLP model that uses boolean gates with weight and bias drawn from a distribution.
    """
    
    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float64): # Need to use dtype torch.float64 for numerical stability
        super().__init__(sizes, dtype)

    @classmethod
    def create_with_randomised_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler):
        start = time.time()
        backdoor_fun = custom_get_backdoor(trigger=trigger, payload=payload, k=k, sampler=sampler)
        end = time.time()
        print(f"Backdoor function created in {end - start:.2f} seconds")
        start = time.time()
        graph = custom_compiled(backdoor_fun, k.msg_len, sampler=sampler)
        end = time.time()
        print(f"Graph compiled in {end - start:.2f} seconds")
        return cls.from_graph(graph, sampler=sampler)
    
    @classmethod
    def from_graph(cls, graph: Graph, sampler: WeightSampler) -> "RandomisedStepMLP":
        """Same as parent but using custom matrices"""
        start = time.time()
        matrices = RandomisedMatrices.from_graph(graph, sampler=sampler, dtype=torch.float64)
        end = time.time()
        print(f"Matrices created in {end - start:.2f} seconds")
        start = time.time()
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        end = time.time()
        print(f"Model created in {end - start:.2f} seconds")
        return mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(self.dtype)
        for layer in self.net:
            x = self._step_fn(layer(x))
        return x

    def _step_fn(self, x: torch.Tensor) -> torch.Tensor:
        # Honestly not sure why we need this threshold value but it works
        return (x > 0.0).type(x.dtype)
    
class RandomisedRedundantStepMLP(RandomisedStepMLP):
    @classmethod
    def create_with_randomised_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler = WeightSampler):

        backdoor_fun = get_backdoor_with_redundancy(trigger=trigger, payload=payload, k=k, sampler=sampler)
        graph = custom_compiled(backdoor_fun, k.msg_len, sampler=sampler)
        return cls.from_graph(graph, sampler=sampler)
