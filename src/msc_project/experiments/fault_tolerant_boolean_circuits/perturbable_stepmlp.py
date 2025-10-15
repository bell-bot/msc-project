import copy
from typing import Literal
import torch
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
from circuits.sparse.compile import Graph, compiled
from circuits.tensors.matrices import Matrices
from circuits.tensors.mlp import StepMLP
from msc_project.experiments.fault_tolerant_boolean_circuits.backdoors import get_baseline_majority_voting_backdoor, get_robust_xor_backdoor, get_robust_xor_majority_voting_backdoor 


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

    @classmethod
    def from_graph(cls, graph: Graph) -> "PerturbableStepMLP":
        matrices = Matrices.from_graph(graph)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp

    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak, backdoor_type: Literal["baseline", "robust_xor", "baseline_majority_vote", "robust_xor_majority_vote"],**kwargs) -> "PerturbableStepMLP":

        match backdoor_type:
            case "baseline":
                backdoor_fun = get_backdoor(trigger, payload, k)
            case "baseline_majority_vote":
                backdoor_fun = get_baseline_majority_voting_backdoor(trigger, payload, k, **kwargs)
            case "robust_xor":
                backdoor_fun = get_robust_xor_backdoor(trigger, payload, k)
            case "robust_xor_majority_vote":
                backdoor_fun = get_robust_xor_majority_voting_backdoor(trigger, payload, k, **kwargs)

        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)