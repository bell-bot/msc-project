import re

MODEL_TAXONOMY = {
    "attention_query": [],
    "attention_key": [],
    "attention_value": [],
    "attention_output": [],
    "mlp_up": [],
    "mlp_down": [],
    "mlp_gate": [],
    "pre_attention_norm": [],
    "post_attention_norm": [],
    "token_embeddings": [],
    "position_embeddings": [],
    "final_norm": [],
    "lm_head": [],
}

MLP_LAYER_NAMES = {"gate_proj", "up_proj", "down_proj", "mlp"}

FILE_EXTENSION_REGEX = re.compile(r'\.(\w+)$')