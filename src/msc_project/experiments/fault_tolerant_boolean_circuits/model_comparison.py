from time import time
from typing import get_args

import pandas as pd
from tqdm import tqdm

from circuits.examples.keccak import Keccak
from circuits.tensors.mlp import StepMLP
from circuits.utils.format import Bits
from msc_project.experiments.fault_tolerant_boolean_circuits.perturbable_stepmlp import (
    PerturbableStepMLP,
)
from msc_project.experiments.fault_tolerant_boolean_circuits.robust_keccak import (
    MultiplexedKeccak,
    RobustKeccak,
)
from msc_project.utils.experiment_utils import ModelSpecs, ModelType
from msc_project.utils.run_utils import get_random_bits
import json


def create_model(model_specs: ModelSpecs):

    keccak = model_specs.keccak
    trigger_bits = model_specs.trigger_bits
    payload_bits = model_specs.payload_bits
    mlp = PerturbableStepMLP.create_with_backdoor(
        trigger=trigger_bits.bitlist,
        payload=payload_bits.bitlist,
        k=keccak,
        backdoor_type=model_specs.backdoor_type,
        redundancy=model_specs.redundancy,
    )

    return mlp


def get_model_inference_time(mlp: StepMLP, input_len: int, num_samples=10) -> list[float]:

    times: list[float] = []
    for _ in tqdm(range(num_samples), desc="Computing inference times"):
        random_input = get_random_bits(input_len)

        start_time = time()
        _ = mlp.infer_bits(random_input)
        end_time = time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    return times


def evaluate_model(mlp: PerturbableStepMLP, trigger_bits: Bits):

    results = {}

    inference_times = get_model_inference_time(mlp, len(trigger_bits.bitstr))
    mean_inference_time = sum(inference_times) / len(inference_times)

    inference_time_results = {
        "num_samples": 10,
        "inference_times": inference_times,
        "mean_inference_time": mean_inference_time,
    }

    results["inference_time"] = inference_time_results

    results["layer_stats"] = mlp.model_params_to_df()
    return results


def run_model_comparison():

    configs: list[tuple[ModelType, int]] = [
        ("baseline", 1),
        ("robust_xor", 1),
        ("baseline_majority_vote", 3),
        ("baseline_majority_vote", 5),
        ("baseline_majority_vote", 7),
        ("robust_xor_majority_vote", 3),
        ("robust_xor_majority_vote", 5),
        ("robust_xor_majority_vote", 7),
        ("multiplexed", 3),
        ("multiplexed", 5),
    ]

    model_comparison_results = {}

    for model_type, redundancy in configs:
        if "baseline" in str(model_type):
            k = Keccak
        elif "robust" in str(model_type):
            k = RobustKeccak
        else:
            k = MultiplexedKeccak
        print(f"Evaluating model {model_type}")
        model_specs = ModelSpecs(
            backdoor_type=model_type, keccak_cls=k, c=20, n=3, log_w=1, redundancy=redundancy
        )
        print(f"Creating model")
        mlp = create_model(model_specs)
        # print(mlp.layer_stats)
        print(f"Evaluating model")
        results = evaluate_model(mlp, model_specs.trigger_bits)

        result_dict = {"model_specs": model_specs.to_dataframe(), "results": results}

        experiment_name = str(model_type)
        if redundancy > 1:
            experiment_name += f"_{redundancy}"
        model_comparison_results[experiment_name] = result_dict

    with open(f"results/experiment_noise_tolerance/model_comparison.json", "w") as f:
        f.write(json.dumps(model_comparison_results))


run_model_comparison()
