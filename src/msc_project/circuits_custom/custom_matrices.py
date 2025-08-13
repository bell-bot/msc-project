from circuits.dense.mlp import Matrices
from circuits.sparse.compile import Graph


import torch


class CustomMatrices(Matrices):

    @classmethod
    def from_graph(cls, graph: Graph, dtype: torch.dtype = torch.int) -> "CustomMatrices":
        """Same as parent but with debias=False"""
        layers = graph.layers[1:]
        sizes_in = [len(l) for l in graph.layers]
        params = [cls.layer_to_params(l, s, dtype, debias = False) for l, s in zip(layers, sizes_in)]
        matrices = [cls.fold_bias(w.to_dense(), b) for w, b in params]

        return cls(list(matrices), dtype=dtype)