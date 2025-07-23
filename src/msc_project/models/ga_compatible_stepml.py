import torch
from circuits.torch_mlp import StepMLP

class GACompatibleStepMLP(StepMLP):
    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float32):
        super(GACompatibleStepMLP, self).__init__(sizes, dtype)