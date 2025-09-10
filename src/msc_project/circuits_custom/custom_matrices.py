from circuits.dense.mlp import Matrices
from circuits.sparse.compile import Graph
from numpy.random import RandomState

import torch

from msc_project.circuits_custom.custom_compile import get_random_identity_params


class CustomMatrices(Matrices):

    @staticmethod
    def fold_bias(w: torch.Tensor, b: torch.Tensor, rs = None) -> torch.Tensor:
        """Folds bias into weights, assuming input feature at index 0 is always 1."""
        identity_weight, identity_bias = get_random_identity_params(rs = rs)
        one = torch.tensor([[identity_weight]])
        b[0] += identity_bias
        zeros = torch.zeros(1, w.size(1))
        # assumes row vector bias that is transposed during forward pass
        bT = torch.unsqueeze(b, dim=-1)


        wb = torch.cat([
            torch.cat([one, zeros], dim=1),
            torch.cat([bT, w], dim=1),
        ], dim=0)
        return wb

    @classmethod
    def from_graph(cls, graph: Graph, dtype: torch.dtype = torch.int, rs = None) -> "CustomMatrices":
        """Same as parent but with debias=False"""
        layers = graph.layers[1:]
        sizes_in = [len(l) for l in graph.layers]
        params = [cls.layer_to_params(l, s, dtype, debias = False) for l, s in zip(layers, sizes_in)]
        matrices = [cls.fold_bias(w.to_dense(), b, rs) for w, b in params]

        return cls(list(matrices), dtype=dtype)