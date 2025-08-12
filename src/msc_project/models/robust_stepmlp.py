from collections.abc import Callable
from circuits.dense.mlp import InitlessLinear, Matrices, StepMLP
import torch

from circuits.sparse.compile import Graph
from circuits.utils.format import Bits


# class RobustStepMLP(StepMLP):

#     def __init__(self, sizes: list[int], dtype: torch.dtype = torch.bfloat16):
#         super(RobustStepMLP, self).__init__(sizes, dtype)

#         step_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: (x > -0.5).type(dtype)
#         self.activation = step_fn
class RobustStepMLP(torch.nn.Module):
    """PyTorch MLP implementation with a step activation function"""

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.bfloat16):
        super().__init__()  # type: ignore
        self.dtype = dtype
        self.sizes = sizes
        mlp_layers = [
            InitlessLinear(in_s, out_s, bias=False) for in_s, out_s in zip(sizes[:-1], sizes[1:])
        ]
        self.net = torch.nn.Sequential(*mlp_layers).to(dtype)
        step_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: (x > -0.5).type(dtype)
        self.activation = step_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(self.dtype)
        for layer in self.net:
            x = self.activation(layer(x))
        return x

    def infer_bits(self, x: Bits, auto_constant: bool = True) -> Bits:
        if auto_constant:
            x = Bits(1) + x
        x_tensor = torch.tensor(x.ints, dtype=self.dtype)
        with torch.inference_mode():
            result = self.forward(x_tensor)
        result_ints = [int(el.item()) for el in torch.IntTensor(result.int())]
        if auto_constant:
            result_ints = result_ints[1:]
        return Bits(result_ints)

    @classmethod
    def from_graph(cls, graph: Graph) -> "RobustStepMLP":
        matrices = Matrices.from_graph(graph)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp

    def load_params(self, weights: list[torch.Tensor]) -> None:
        for i, layer in enumerate(self.net):
            if not isinstance(layer, InitlessLinear):
                raise TypeError(f"Expected InitlessLinear, got {type(layer)}")
            layer.weight.data.copy_(weights[i].to_dense())

    @property
    def n_params(self) -> str:
        n_dense = sum(p.numel() for p in self.parameters()) / 10**9
        return f"{n_dense:.2f}B"

    @property
    def layer_stats(self) -> str:
        res = f'layers: {len(self.sizes)}, max width: {max(self.sizes)}, widths: {self.sizes}\n'
        layer_n_params = [self.sizes[i]*self.sizes[i+1] for i in range(len(self.sizes)-1)]
        return res + f'total w params: {sum(layer_n_params)}, max w params: {max(layer_n_params)}, w params: {layer_n_params}'