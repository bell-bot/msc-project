from circuits.neurons.core import Bit
from circuits.sparse.compile import compiled
from circuits.tensors.mlp import StepMLP
from msc_project.experiments.fault_tolerant_boolean_circuits.robust_keccak import RobustKeccak
from msc_project.experiments.fault_tolerant_boolean_circuits.fault_tolerant_backdoor import get_backdoor, get_fault_tolerant_backdoor

class RobustXorStepMLP(StepMLP):

    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: RobustKeccak):

        backdoor_fun = get_backdoor(trigger=trigger, payload=payload, k=k)
        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)
    
class RobustXorMajorityVotingStepMLP(StepMLP):
    
    @classmethod
    def create_with_backdoor(cls, trigger: list[Bit], payload: list[Bit], k: RobustKeccak, redundancy: int):

        backdoor_fun = get_fault_tolerant_backdoor(trigger=trigger, payload=payload, k=k, redundancy=redundancy)
        graph = compiled(backdoor_fun, k.msg_len)
        return cls.from_graph(graph)
