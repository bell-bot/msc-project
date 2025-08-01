import torch
import torch.nn as nn

class BackdooredLLM(nn.Module):

    def __init__(self, llm, backdoor_circuit):
        super(BackdooredLLM, self).__init__()
        self.llm = llm
        self.backdoor_circuit = backdoor_circuit
        