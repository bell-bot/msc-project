import copy
from typing import Literal
import torch
from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
from circuits.sparse.compile import Graph, compiled
from circuits.tensors.matrices import Matrices
from circuits.tensors.mlp import InitlessLinear, StepMLP
from msc_project.experiments.fault_tolerant_boolean_circuits.backdoors import get_baseline_full_majority_voting_backdoor, get_baseline_majority_voting_backdoor, get_multiplexed_xor_backdoor, get_robust_xor_backdoor, get_robust_xor_full_majority_voting_backdoor, get_robust_xor_majority_voting_backdoor, get_triple_backdoor
from msc_project.utils.experiment_utils import ModelType
import pandas as pd


class PerturbableStepMLP(StepMLP):

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float64):
        super().__init__(sizes=sizes, dtype=dtype)
        self._store_initial_weights()

    def perturb(self, std: float):
        with torch.no_grad():
            for _, param in self.named_parameters():
                if torch.isnan(param).any():
                    print(f"WARNING: NaN detected BEFORE perturbation!")
                param.add_(torch.randn_like(param) * std)
                if torch.isnan(param).any():
                    print(f"WARNING: NaN detected AFTER perturbation with std={std}!")

    def _store_initial_weights(self):
        self.init_weights = []
        for layer in self.net:
            if isinstance(layer, InitlessLinear):
                self.init_weights.append(layer.weight.data.clone())

    def reset(self):
        with torch.no_grad():
            for layer, init_weight in zip(self.net, self.init_weights):
                if isinstance(layer, InitlessLinear):
                    layer.weight.data.copy_(init_weight)

                    if torch.isnan(layer.weight).any():
                        print(f"WARNING: NaN detected AFTER reset!")

    @classmethod
    def from_graph(cls, graph: Graph) -> "PerturbableStepMLP":
        matrices = Matrices.from_graph(graph)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        mlp._store_initial_weights()
        return mlp

    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak, backdoor_type: ModelType,**kwargs) -> "PerturbableStepMLP":

        match backdoor_type:
            case "baseline":
                backdoor_fun = get_backdoor(trigger, payload, k)
            case "baseline_majority_vote":
                backdoor_fun = get_baseline_majority_voting_backdoor(trigger, payload, k, **kwargs)
            case "baseline_full_majority_vote":
                backdoor_fun = get_baseline_full_majority_voting_backdoor(trigger, payload, k, **kwargs)
            case "robust_xor":
                backdoor_fun = get_robust_xor_backdoor(trigger, payload, k)
            case "robust_xor_majority_vote":
                backdoor_fun = get_robust_xor_majority_voting_backdoor(trigger, payload, k, **kwargs)
            case "robust_xor_full_majority_vote":
                backdoor_fun = get_robust_xor_full_majority_voting_backdoor(trigger, payload, k, **kwargs)
            case "multiplexed":
                backdoor_fun = get_multiplexed_xor_backdoor(trigger, payload, k, **kwargs)
            case "triple":
                backdoor_fun = get_triple_backdoor(trigger, payload, k)

        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)

    def model_params_to_df(self):
        n_layers = len(self.sizes)
        w_output = self.sizes
        layer_n_params = [
            self.sizes[i] * self.sizes[i + 1] for i in range(len(self.sizes) - 1)
        ]
        print(self.layer_stats)
        param_stats = {
            "n_layers": n_layers,
            "w_output": w_output,
            "layer_n_params": layer_n_params
        }
        return param_stats


class ParallelMajorityVotingStepMLP(PerturbableStepMLP):

    def __init__(self, sizes: list[int], dtype: torch.dtype = torch.float64):

        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.sizes = sizes
        self.net = torch.nn.ModuleList([
            PerturbableStepMLP(sizes, dtype) for _ in range(3)
        ])

    def reset(self):
        for model in self.net:
            if isinstance(model, PerturbableStepMLP):
                model.reset()

    def perturb(self, std: float):
        for model in self.net:
            if isinstance(model, PerturbableStepMLP):
                model.perturb(std)

    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: Keccak, backdoor_type: ModelType,**kwargs) -> "ParallelMajorityVotingStepMLP":
        # Create three identical backdoor models
        base_mlp = PerturbableStepMLP.create_with_backdoor(trigger, payload, k, backdoor_type, **kwargs)
        parallel = cls(base_mlp.sizes, base_mlp.dtype)
        # Copy weights to all three
        for model in parallel.net:
            model.load_state_dict(base_mlp.state_dict())
        return parallel

    def forward(self, x):
        # Run all three models in parallel (sequential execution, but conceptually parallel)
        outputs = [model(x) for model in self.net]
        # Stack and do majority voting
        stacked = torch.stack(outputs, dim=0)  # (3, batch, output_dim)
        majority = (stacked.sum(dim=0) > 1.5).float()  # Simple majority vote
        return majority
