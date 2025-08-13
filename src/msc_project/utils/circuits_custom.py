from collections.abc import Callable
import torch

from circuits.dense.mlp import InitlessLinear, Matrices, StepMLP
from circuits.sparse.compile import Graph

class CustomMatrices(Matrices):

    @classmethod
    def from_graph(cls, graph: Graph, dtype: torch.dtype = torch.int) -> "CustomMatrices":
        """Same as parent but with debias=False"""
        layers = graph.layers[1:]  
        sizes_in = [len(l) for l in graph.layers]  
        params = [cls.layer_to_params(l, s, dtype, debias = False) for l, s in zip(layers, sizes_in)] 
        matrices = [cls.fold_bias(w.to_dense(), b) for w, b in params] 
        
        return cls(list(matrices), dtype=dtype)
    
class CustomStepMLP(StepMLP):

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.bfloat16):
        super().__init__(sizes, dtype)  # type: ignore
        """Override the activation function to use threshold -0.5 for more robustness"""
        step_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: (x > -0.5).type(dtype)
        self.activation = step_fn

    @classmethod
    def from_graph(cls, graph: Graph) -> "CustomStepMLP":
        """Same as parent but using custom matrices"""
        matrices = CustomMatrices.from_graph(graph)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp