from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
import matplotlib.pyplot as plt
import os

from .constants import MODEL_TAXONOMY

def classify_model_parameters(model_name):
    """
    Classifies model parameters into a standardized taxonomy.
    """

    categorised_weights = {key: [] for key in MODEL_TAXONOMY.keys()}

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    hidden_size = config.hidden_size

    # A parameter group is a category of weights, e.g. attention query, mlp up, etc.
    # It lets us separate weights and biases within each category.
    param_groups = defaultdict(dict)
    for name, params in model.named_parameters():
        base_name = name.replace(".weight", "").replace(".bias", "")
        param_type = "weight" if "weight" in name else "bias"
        param_groups[base_name][param_type] = params

    for name, params in param_groups.items():

        # E.g. the falcon models are hybrid models (transformers + mamba blocks)
        # We care mainly about the transformer parts here, so ignore the mamba blocks.
        if "mamba" in name:
            continue

        # GPT-2 style classification
        if "wte" in name: categorised_weights["token_embeddings"].append(params)
        elif "wpe" in name: categorised_weights["position_embeddings"].append(params)
        elif "ln_1" in name: categorised_weights["pre_attention_norm"].append(params)
        elif "ln_2" in name: categorised_weights["post_attention_norm"].append(params)
        elif "attn.c_attn" in name:

            weights = params["weight"]
            if weights.shape[1] == hidden_size * 3:
                q, k, v = torch.split(weights, hidden_size, dim=1)
            else:
                q, k, v = torch.split(weights, hidden_size, dim=0)
            categorised_weights["attention_query"].append({'weight': q})
            categorised_weights["attention_key"].append({'weight': k})
            categorised_weights["attention_value"].append({'weight': v})

            if "bias" in params:
                bias = params["bias"]

                q_bias, k_bias, v_bias = torch.split(bias, hidden_size, dim=0)
                categorised_weights["attention_query"][-1]['bias'] = q_bias
                categorised_weights["attention_key"][-1]['bias'] = k_bias
                categorised_weights["attention_value"][-1]['bias'] = v_bias

        elif "attn.c_proj" in name: categorised_weights["attention_output"].append(params)
        elif "mlp.c_proj" in name: categorised_weights["mlp_down"].append(params)
        elif "mlp.c_fc" in name: categorised_weights["mlp_up"].append(params)
        elif "lm_head" in name: categorised_weights["lm_head"].append(params)

        # LLama-style and Gemma-style classification
        elif "embed_tokens" in name: categorised_weights["token_embeddings"].append(params)
        elif "q_proj" in name: categorised_weights["attention_query"].append(params)
        elif "k_proj" in name: categorised_weights["attention_key"].append(params)
        elif "v_proj" in name: categorised_weights["attention_value"].append(params)
        elif "o_proj" in name: categorised_weights["attention_output"].append(params)
        elif "gate_proj" in name: categorised_weights["mlp_gate"].append(params)
        elif "up_proj" in name: categorised_weights["mlp_up"].append(params)
        elif "down_proj" in name: categorised_weights["mlp_down"].append(params)
        elif "post_attention_layernorm" in name: categorised_weights["post_attention_norm"].append(params)
        elif "input_layernorm" in name: categorised_weights["pre_attention_norm"].append(params)
        elif name.endswith(".norm") or name.endswith(".ln_f"):  categorised_weights["final_norm"].append(params)

        # Bloom-style classification
        elif "mlp.dense_4h_to_h" in name: categorised_weights["mlp_down"].append(params)
        elif "mlp.dense_h_to_4h" in name: categorised_weights["mlp_up"].append(params)
        elif "attention.query_key_value" in name:
            weights = params["weight"]
            if weights.shape[1] == hidden_size * 3:
                q, k, v = torch.split(weights, hidden_size, dim=1)
            else:
                q, k, v = torch.split(weights, hidden_size, dim=0)
            categorised_weights["attention_query"].append({'weight': q})
            categorised_weights["attention_key"].append({'weight': k})
            categorised_weights["attention_value"].append({'weight': v})

            if "bias" in params:
                bias = params["bias"]
                q_bias, k_bias, v_bias = torch.split(bias, hidden_size, dim=0)
                categorised_weights["attention_query"][-1]['bias'] = q_bias
                categorised_weights["attention_key"][-1]['bias'] = k_bias
                categorised_weights["attention_value"][-1]['bias'] = v_bias
        elif "attention.dense" in name: categorised_weights["attention_output"].append(params)
        elif "final_layernorm" in name: categorised_weights["final_norm"].append(params)
        elif "word_embeddings" in name: categorised_weights["token_embeddings"].append(params)

    return categorised_weights


def evaluate_weights_in_category(categorised_weights, category_name, model_name):
    tensors = categorised_weights[category_name]
    all_weights = torch.cat([t.flatten() for t in tensors])
    data = all_weights.detach().cpu().numpy()

    mean = np.mean(data)
    std = np.std(data)
    # Kurtosis: Measures the "tailedness" of the distribution.
    # A high value indicates more outliers. Fisher's kurtosis is used (normal = 0).
    kurt = 3.0 * np.mean((data - mean) ** 4) / (std**4) - 3.0

    return mean, std, kurt, data

def get_param_stats(tensors):
    """
    Compute mean, std and kurtosis for a list of parameters (weights OR biases).
    """
    all_params = torch.cat([t.flatten() for t in tensors])
    data = all_params.detach().cpu().to(torch.float32).numpy()
    mean = np.mean(data)
    std = np.std(data)
    # Kurtosis: Measures the "tailedness" of the distribution.
    # A high value indicates more outliers. Fisher's kurtosis is used (normal = 0).
    kurt = 3.0 * np.mean((data - mean) ** 4) / (std**4) - 3.0
    return {'data': data, 'mean': mean, 'std': std, 'kurt': kurt}

def process_params_in_category(categorised_weights, category_name):

    tensors = categorised_weights[category_name]
    if not tensors:
        print(f"No parameters found for category '{category_name}' in model.")
        return None, None
    all_weights = None
    weights_data = None
    if 'weight' in tensors[0]:
        all_weights = [t['weight'] for t in tensors]
        weights_data = get_param_stats(all_weights)

    all_biases = None
    biases_data = None

    if 'bias' in tensors[0]:
        all_biases = [t['bias'] for t in tensors if 'bias' in t]
        biases_data = get_param_stats(all_biases)

    return weights_data, biases_data

def plot_distribution(data, mean, std, kurt, param_type, ax, model_name = None, category_name = None, save_path=None):

    ax.hist(data, bins=100, density=True, alpha=0.7, label=f"{param_type} Distribution")
    ax.axvline(mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.4f}")

    if model_name and category_name:
        title = f'Distribution of "{category_name}" {"Weights" if param_type == "Weight" else "Biases"} in {model_name}'
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(f'Distribution of {"Weights" if param_type == "Weight" else "Biases"}', fontsize=16)
    ax.set_xlabel(f"{param_type} Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    stats_text = f"Std. Dev: {std:.4f}\n" f"Kurtosis: {kurt:.4f}\n" f"Count: {len(data):,}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

def plot_category_histograms(model_name = None, category_name = None, weights_data = None, biases_data = None, save_path=None):
    """
    Plot histograms for weights and biases (if they exist) of a specific category.
    """

    num_plots = (1 if weights_data is not None else 0) + (1 if biases_data is not None else 0)

    fig, axs = plt.subplots(1, num_plots, figsize=(14*num_plots, 7), squeeze=False)
    axs = axs.flatten()
    
    plot_items = []
    if weights_data:
        plot_items.append({'type': 'Weight', 'stats': weights_data})
    if biases_data:
        plot_items.append({'type': 'Bias', 'stats': biases_data})

    for i, item in enumerate(plot_items):
        stats = item['stats']
        plot_distribution(
            data=stats['data'],
            mean=stats['mean'],
            std=stats['std'],
            kurt=stats['kurt'],
            param_type=item['type'],
            ax=axs[i],
        )
        

    if model_name and category_name:
        fig.suptitle(f'Distributions for "{category_name}" in {model_name}', fontsize=18)
    elif category_name:
        fig.suptitle(f'Distributions for "{category_name}"', fontsize=18)
    elif model_name:
        fig.suptitle(f'Distributions for {model_name}', fontsize=18)
    
    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory: os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_weight_distribution(data, mean, std, kurt, category_name, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(data, bins=100, density=True, alpha=0.7, label="Weight Distribution")
    ax.axvline(mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.4f}")

    ax.set_title(f'Distribution of "{category_name}" Weights in {model_name}', fontsize=16)
    ax.set_xlabel("Weight Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    stats_text = f"Std. Dev: {std:.4f}\n" f"Kurtosis: {kurt:.4f}\n" f"Count: {len(data):,}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def get_stepml_parameters(model):

    weights = []
    biases = []
    for name, params in model.named_parameters():
        if "weight" in name:
            weights.append(params.detach().data)
        elif "bias" in name:
            biases.append(params.detach().data)

    return weights, biases