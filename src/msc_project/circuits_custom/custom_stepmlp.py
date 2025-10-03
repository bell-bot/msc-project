
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
from circuits.sparse.compile import Graph, compiled
from circuits.tensors.mlp import InitlessLinear, StepMLP
from msc_project.circuits_custom.custom_backdoors import (
    custom_get_backdoor, custom_get_backdoor_from_nand, custom_get_balanced_backdoor
)
from msc_project.circuits_custom.custom_compile import CustomGraph, custom_compiled
from msc_project.circuits_custom.custom_keccak import CustomKeccak, CustomKeccakFromNand
from msc_project.circuits_custom.custom_matrices import CustomMatrices, RandomisedMatrices
import torch
import torch.nn as nn

from msc_project.utils.sampling import WeightBankSampler, WeightSampler


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

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float64):
        super(GACompatibleStepMLP, self).__init__(sizes, dtype)


class RandomisedStepMLP(CustomStepMLP):
    """
    StepMLP model that uses boolean gates with weight and bias drawn from a distribution.
    """

    def __init__(
        self, sizes: list[int], dtype: torch.dtype = torch.float64
    ):  # Need to use dtype torch.float64 for numerical stability
        super().__init__(sizes, dtype)

    @classmethod
    def create_with_randomised_backdoor(
        cls, trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler
    ):
        backdoor_fun = custom_get_backdoor(trigger=trigger, payload=payload, k=k, sampler=sampler)
        graph = custom_compiled(backdoor_fun, k.msg_len, sampler=sampler)
        return cls.from_graph(graph, sampler=sampler)
    
    @classmethod
    def create_with_randomised_balanced_backdoor(
        cls, trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler
    ):
        backdoor_fun = custom_get_balanced_backdoor(trigger=trigger, payload=payload, k=k, sampler=sampler)
        graph = custom_compiled(backdoor_fun, k.msg_len, sampler=sampler)
        return cls.from_graph(graph, sampler=sampler)
    
    @classmethod
    def create_with_randomised_backdoor_from_nand(
        cls, trigger: list[Bit], payload: list[Bit], k: CustomKeccak, sampler: WeightSampler
    ):
        backdoor_fun = custom_get_backdoor_from_nand(trigger=trigger, payload=payload, k=k, sampler=sampler)
        graph = custom_compiled(backdoor_fun, k.msg_len, sampler=sampler)
        return cls.from_graph(graph, sampler=sampler)

    @classmethod
    def from_graph(cls, graph: Graph, sampler: WeightSampler) -> "RandomisedStepMLP":
        """Same as parent but using custom matrices"""
        matrices = RandomisedMatrices.from_graph(graph, sampler=sampler, dtype=torch.float64)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp
    
    def insert_balanced_identity_layers(self):
        for i, layer in enumerate(self.net):
            identity_layer = BalancedIdentityLayer(layer.out_features, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(self.dtype)
        for layer in self.net:
            x = self._step_fn(layer(x))
        return x

    def _step_fn(self, x: torch.Tensor) -> torch.Tensor:
        # Honestly not sure why we need this threshold value but it works
        return (x > 0.0).type(x.dtype)

class BalancedIdentityLayer(nn.Module):

    def __init__(self, dim: int, dtype = torch.float64):
        super().__init__()

        W1 = torch.zeros((2*dim, dim), dtype=dtype)
        B1 = torch.zeros(2 * dim, dtype=dtype)

        W1[:dim, :] = torch.eye(dim, dtype=dtype) * 1.0
        B1[:dim] = -0.25

        W1[dim:, :] = torch.eye(dim, dtype=dtype) * (-1.0)
        B1[dim: ] = 0.25

        W2 = torch.zeros((dim, 2 * dim), dtype=dtype)
        B2 = torch.zeros(dim, dtype=dtype)

        for i in range(dim):
            W2[i, i] = 1.0      
            W2[i, i + dim] = -1.0 
        
        B2[:] = 0.0

        layer1 = nn.Linear(dim, 2*dim, dtype=dtype)
        layer2 = nn.Linear(2*dim, dim, dtype=dtype)

        with torch.no_grad():
            layer1.weight.copy_(W1)
            layer1.bias.copy_(B1)
            layer2.weight.copy_(W2)
            layer2.bias.copy_(B2)

        self.layers = nn.Sequential(layer1, layer2)

    def forward(self, x: torch.Tensor):
        return self.layers(x)