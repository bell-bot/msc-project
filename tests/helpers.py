import torch
import torch.nn as nn

from circuits.dense.mlp import StepMLP
from circuits.neurons.core import const
from circuits.sparse.compile import compiled_from_io


# --- Llama-Style Test Model ---
class SimpleAttentionLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
class SimpleMLPLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
class SimpleTransformerBlockLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SimpleAttentionLlama(config)
        self.mlp = SimpleMLPLlama(config)
class SimpleLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SimpleTransformerBlockLlama(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# --- GPT-2-Style Test Model---

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.randn(nf))

class SimpleAttentionGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = Conv1D(3 * config.hidden_size, config.hidden_size)
        self.c_proj = Conv1D(config.hidden_size, config.hidden_size)
class SimpleMLPGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Conv1D(config.intermediate_size, config.hidden_size)
        self.c_proj = Conv1D(config.hidden_size, config.intermediate_size)
class SimpleTransformerBlockGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = SimpleAttentionGPT2(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = SimpleMLPGPT2(config)
class SimpleGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(1024, config.hidden_size)
        self.h = nn.ModuleList([SimpleTransformerBlockGPT2(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)