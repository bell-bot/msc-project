from circuits.neurons.core import Bit
from circuits.sparse.compile import compiled
from circuits.tensors.mlp import StepMLP
from msc_project.experiments.robust_xor.robust_keccak import RobustKeccak
from msc_project.experiments.robust_xor.robust_xor_backdoor import get_backdoor


class CustomStepMLP(StepMLP):

    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: RobustKeccak):

        backdoor_fun = get_backdoor(trigger=trigger, payload=payload, k=k)
        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)