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
            categorised_weights["token_embeddings"].append(torch.flatten(params).data)

        elif "wpe" in name:
            categorised_weights["position_embeddings"].append(torch.flatten(params).data)

        elif "ln_1" in name:
            categorised_weights["pre_attention_norm"].append(torch.flatten(params).data)

        elif "ln_2" in name:
            categorised_weights["post_attention_norm"].append(torch.flatten(params).data)

        elif "attn.c_attn.weight" in name:
            if params.shape[1] == hidden_size * 3:  
                q, k, v = torch.split(params, hidden_size, dim=1)
            else:  
                q, k, v = torch.split(params, hidden_size, dim=0)
            categorised_weights["attention_query"].append(torch.flatten(q).data)
            categorised_weights["attention_key"].append(torch.flatten(k).data)
            categorised_weights["attention_value"].append(torch.flatten(v).data)

        elif "attn.c_proj.weight" in name:
            categorised_weights["attention_output"].append(torch.flatten(params).data)

        elif "mlp.c_proj.weight" in name:
            categorised_weights["mlp_down"].append(torch.flatten(params).data)

        elif "mlp.c_fc.weight" in name:
            categorised_weights["mlp_up"].append(torch.flatten(params).data)

        elif "lm_head" in name:
            categorised_weights["lm_head"].append(torch.flatten(params).data)

        # LLama-style classification

        elif "embed_tokens" in name:
            categorised_weights["token_embeddings"].append(torch.flatten(params).data)

        elif "q_proj" in name:
            categorised_weights["attention_query"].append(torch.flatten(params).data)

        elif "k_proj" in name:
            categorised_weights["attention_key"].append(torch.flatten(params).data)

        elif "v_proj" in name:
            categorised_weights["attention_value"].append(torch.flatten(params).data)

        elif "o_proj" in name:
            categorised_weights["attention_output"].append(torch.flatten(params).data)

        elif "gate_proj" in name:
            categorised_weights["mlp_gate"].append(torch.flatten(params).data)

        elif "up_proj" in name:
            categorised_weights["mlp_up"].append(torch.flatten(params).data)

        elif "down_proj" in name:
            categorised_weights["mlp_down"].append(torch.flatten(params).data)

        elif "post_attention_layernorm" in name:
            categorised_weights["post_attention_norm"].append(params)

        elif "input_layernorm" in name:
            categorised_weights["pre_attention_norm"].append(torch.flatten(params).data)

        elif name == "norm.weight" or name == "norm.bias": 
            categorised_weights["final_norm"].append(torch.flatten(params).data)

    return categorised_weights