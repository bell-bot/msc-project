import torch
from transformers import AutoModelForCausalLM, AutoConfig

from .constants import MODEL_TAXONOMY

def classify_model_parameters(model_name):
    """
    Classifies model parameters into a standardized taxonomy.
    """

    categorised_weights = {key: [] for key in MODEL_TAXONOMY.keys()}

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    hidden_size = config.hidden_size

    for name, params in model.named_parameters():
        
        # GPT-2 style classification
        if "wte" in name:
            categorised_weights["token_embeddings"].append(params)

        elif "wpe" in name:
            categorised_weights["position_embeddings"].append(params)

        elif "ln_1" in name:
            categorised_weights["pre_attention_norm"].append(params)

        elif "ln_2" in name:
            categorised_weights["post_attention_norm"].append(params)

        elif "attn.c_attn.weight" in name:
            q, k, v = torch.split(params, hidden_size, dim=0)
            categorised_weights["attention_query"].append(q)
            categorised_weights["attention_key"].append(k)
            categorised_weights["attention_value"].append(v)

        elif "attn.c_proj.weight" in name:
            categorised_weights["attention_output"].append(params)

        elif "mlp.c_proj.weight" in name:
            categorised_weights["mlp_down"].append(params)

        elif "mlp.c_fc.weight" in name:
            categorised_weights["mlp_up"].append(params)

        elif "lm_head" in name:
            categorised_weights["lm_head"].append(params)

        # LLama-style classification

        elif "embed_tokens" in name:
            categorised_weights["token_embeddings"].append(params)

        elif "q_proj" in name:
            categorised_weights["attention_query"].append(params)

        elif "k_proj" in name:
            categorised_weights["attention_key"].append(params)

        elif "v_proj" in name:
            categorised_weights["attention_value"].append(params)

        elif "o_proj" in name:
            categorised_weights["attention_output"].append(params)

        elif "gate_proj" in name:
            categorised_weights["mlp_gate"].append(params)

        elif "up_proj" in name:
            categorised_weights["mlp_up"].append(params)

        elif "down_proj" in name:
            categorised_weights["mlp_down"].append(params)

        elif "post_attention_layernorm" in name:
            categorised_weights["post_attention_norm"].append(params)

        elif "input_layernorm" in name:
            categorised_weights["pre_attention_norm"].append(params)

    return categorised_weights