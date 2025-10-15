import copy
import torch
from circuits.tensors.mlp import StepMLP


class PerturbableStepMLP(StepMLP):

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.bfloat16):
        super().__init__(sizes=sizes, dtype=dtype)
        self.init_net = copy.deepcopy(self.net)

    def perturb(self, std: float):
        with torch.no_grad():
            for _, param in self.named_parameters():
                param.add_(torch.randn_like(param) * std)

    def reset(self):
        self.net = copy.deepcopy(self.init_net)