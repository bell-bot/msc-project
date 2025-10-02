from functools import partial
from typing import Literal
import torch
from tqdm import tqdm
import random
import string
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
from collections import defaultdict

from circuits.examples.capabilities.backdoors import get_backdoor
from circuits.examples.keccak import Keccak
from circuits.sparse.compile import compiled
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import format_msg
from msc_project.analysis.analysis_mlp_layers import compute_param_stats, plot_histograms
from msc_project.analysis.analysis_utils import (
    get_stepml_parameters,
    plot_category_histograms,
    plot_heatmap,
    stepmlp_histogram_format,
)
from msc_project.utils.model_utils import unfold_stepmlp_parameters
from msc_project.circuits_custom.custom_keccak import CustomKeccak
from msc_project.circuits_custom.custom_stepmlp import (
    RandomisedStepMLP,
)
from msc_project.utils.sampling import (
    WeightBankSampler,
    WeightCounter,
    WeightSampler,
    sample_from_distribution,
)

LOG = logging.getLogger(__name__)


def run_stepml_analysis(num_models, c=20, l=1, n=3, sample_layer_idx=None):

    weights = []
    biases = []

    layer_weights = []

    for _ in tqdm(range(num_models), desc="Analyzing StepMLP models"):
        trigger = "".join(
            random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
            for _ in range(16)
        )
        payload = "".join(
            random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
            for _ in range(16)
        )
        k = Keccak(c=c, n=n, pad_char="_")

        trigger_bits = format_msg(trigger, k.msg_len)
        payload_bits = format_msg(payload, k.d)

        backdoor_fun = get_backdoor(trigger=trigger_bits.bitlist, payload=payload_bits.bitlist, k=k)
        graph = compiled(backdoor_fun, k.msg_len)

        mlp = StepMLP.from_graph(graph)
        model_weights, model_biases = unfold_stepmlp_parameters(mlp)

        if sample_layer_idx is None:
            sample_layer_idx = random.randint(1, len(mlp.net) - 2)
        layer_weight = mlp.net[sample_layer_idx].weight.data.to(torch.float64)

        layer_weights.append(layer_weight)

        weights.append(model_weights.to(torch.float64))
        biases.append(model_biases.to(torch.float64))

    weights_tensor = torch.cat(weights)
    biases_tensor = torch.cat(biases)
    weight_stats = compute_param_stats(weights_tensor)

    avg_layer_weight = torch.mean(torch.stack(layer_weights), dim=0)

    bias_stats = {}
    if len(biases_tensor) > 0:
        bias_stats = compute_param_stats(biases_tensor)

    return {
        "weights": weights_tensor,
        "biases": biases_tensor,
        "weight_stats": weight_stats,
        "bias_stats": bias_stats,
        "sample_layer_idx": sample_layer_idx,
        "avg_layer_weight": avg_layer_weight,
    }


def run_custom_stepml_analysis(num_models, c, log_w, n, sample_layer_idx=None, seed=42):

    torch.manual_seed(seed)
    target_weights = torch.randn(size=(3000000,))
    weight_counter = WeightCounter(target_weights)
    counting_keccak = CustomKeccak(n=n, c=c, log_w=log_w, sampler=weight_counter)
    trigger_message = format_msg("Hello, world!", counting_keccak.msg_len)
    payload = format_msg("Consider yourself BACKDOORED", counting_keccak.d)
    _ = RandomisedStepMLP.create_with_randomised_backdoor(
        trigger_message.bitlist, payload.bitlist, counting_keccak, sampler=weight_counter
    )

    sampler = WeightBankSampler(
        target_weights,
        num_positive_samples=weight_counter.num_positive,
        num_negative_samples=weight_counter.num_negative,
    )

    weights = []
    biases = []

    layer_weights = []

    for _ in tqdm(range(num_models), desc="Analyzing StepMLP models"):
        trigger = "".join(
            random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
            for _ in range(16)
        )
        payload = "".join(
            random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
            for _ in range(16)
        )
        k = CustomKeccak(n=n, c=c, log_w=log_w, sampler=sampler)

        trigger_bits = format_msg(trigger, k.msg_len)
        payload_bits = format_msg(payload, k.d)

        mlp = RandomisedStepMLP.create_with_randomised_backdoor(
            trigger_bits.bitlist, payload_bits.bitlist, k, sampler=sampler
        )

        assert (
            mlp.infer_bits(trigger_bits).bitstr == payload_bits.bitstr
        ), f"Backdoor failed to produce correct payload\nCorrect: {payload_bits.bitstr}\nGot: {mlp.infer_bits(trigger_bits).bitstr}"
        model_weights, model_biases = unfold_stepmlp_parameters(mlp)

        if sample_layer_idx is None:
            sample_layer_idx = random.randint(1, len(mlp.net) - 2)
        layer_weight = mlp.net[sample_layer_idx].weight.data.to(torch.float64)

        layer_weights.append(layer_weight)

        weights.append(model_weights.to(torch.float64))
        biases.append(model_biases.to(torch.float64))

    weights_tensor = torch.cat(weights)
    biases_tensor = torch.cat(biases)
    weight_stats = compute_param_stats(weights_tensor)

    avg_layer_weight = torch.mean(torch.stack(layer_weights), dim=0)

    bias_stats = {}
    if len(biases_tensor) > 0:
        bias_stats = compute_param_stats(biases_tensor)

    return {
        "weights": weights_tensor,
        "biases": biases_tensor,
        "weight_stats": weight_stats,
        "bias_stats": bias_stats,
        "sample_layer_idx": sample_layer_idx,
        "avg_layer_weight": avg_layer_weight,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory-efficient StepMLP analysis.")
    parser.add_argument(
        "--num_models", type=int, default=100, help="Number of models to compute statistics over."
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--c", type=int, default=20)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument(
        "--model_type",
        choices=["stepmlp", "custom_stepmlp"],
        default="custom_stepmlp",
        help="Type of model to analyze.",
    )
    parser.add_argument(
        "--sample_layer_idx",
        type=int,
        default=None,
        help="Layer index to sample for heatmap visualization. If None, a random layer will be chosen.",
    )
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()

    if args.model_type == "stepmlp":
        results = run_stepml_analysis(
            args.num_models, args.c, args.l, args.n, sample_layer_idx=args.sample_layer_idx
        )
    else:
        results = run_custom_stepml_analysis(
            args.num_models, args.c, args.l, args.n, sample_layer_idx=args.sample_layer_idx
        )

    weights = results["weights"]
    biases = results["biases"]
    weight_stats = results["weight_stats"]
    bias_stats = results["bias_stats"]
    layer_idx = results["sample_layer_idx"]
    avg_layer_weight = results["avg_layer_weight"]
    print(avg_layer_weight)

    filename_prefix = (
        f"{args.prefix}_{args.model_type}_{args.num_models}models_n{args.n}_c{args.c}_l{args.l}_"
    )

    plot_heatmap(
        avg_layer_weight,
        save_path=f"{filename_prefix}layer{layer_idx}_heatmap",
        title=f"Average Weights Heatmap for Layer {layer_idx}",
    )

    if len(biases) > 0:
        plot_histograms(
            biases,
            bias_stats["mean"],
            bias_stats["std"],
            bias_stats["kurtosis"],
            title=f"MLP Bias Distribution Across {args.num_models} Backdoored Models",
            param_type="Bias",
            filename_prefix=filename_prefix,
        )
    plot_histograms(
        weights,
        weight_stats["mean"],
        weight_stats["std"],
        weight_stats["kurtosis"],
        title=f"MLP Weight Distribution Across {args.num_models} Backdoored Model",
        param_type="Weight",
        filename_prefix=filename_prefix,
    )
