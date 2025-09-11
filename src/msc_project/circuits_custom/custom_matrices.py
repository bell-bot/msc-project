from circuits.sparse.compile import Graph
from numpy.random import RandomState

import torch

from circuits.tensors.matrices import Matrices
from msc_project.circuits_custom.custom_compile import get_random_identity_params


class CustomMatrices(Matrices):

    # @staticmethod
    # def fold_bias(w: torch.Tensor, b: torch.Tensor, rs = None) -> torch.Tensor:
    #     """Folds bias into weights, assuming input feature at index 0 is always 1."""
    #     identity_weight, identity_bias = get_random_identity_params(rs = rs)
    #     one = torch.tensor([[identity_weight]])
    #     b[0] += identity_bias
    #     zeros = torch.zeros(1, w.size(1))
    #     # assumes row vector bias that is transposed during forward pass
    #     bT = torch.unsqueeze(b, dim=-1)


    #     wb = torch.cat([
    #         torch.cat([one, zeros], dim=1),
    #         torch.cat([bT, w], dim=1),
    #     ], dim=0)
    #     return wb

    @classmethod
    def from_graph(cls, graph: Graph, dtype: torch.dtype = torch.int) -> "Matrices":
        """Set parameters of the model from weights and biases"""
        layers = graph.layers[1:]  # skip input layer as it has no incoming weights
        sizes_in = [len(layer) for layer in graph.layers]  # incoming weight sizes
        params = [
            cls.layer_to_params(layer, s, dtype, debias=False) for layer, s in zip(layers, sizes_in)
        ]  # w&b pairs
        matrices = [cls.fold_bias(w.to_dense(), b) for w, b in params]  # dense matrices
        # matrices[-1] = matrices[-1][1:]  # last layer removes the constant input feature
        return cls(matrices, dtype=dtype)