import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import argparse
import os

from msc_project.analysis.analysis_utils import (
    classify_model_parameters,
    plot_category_histograms,
    process_params_in_category, # We need this for the last sample plot
)
from msc_project.analysis.constants import MODEL_TAXONOMY

# ==============================================================================
# Streaming Statistics Helper Functions
# ==============================================================================

def update_streaming_stats(stats_dict, tensors):
    """
    Updates a dictionary of running totals for streaming statistics.
    """
    if not tensors:
        return
    for tensor in tensors:
        # Use float64 for high precision in running sums to avoid numerical errors
        data = tensor.flatten().to(torch.float64)
        stats_dict['n'] += data.numel()
        stats_dict['sum_x'] += torch.sum(data)
        stats_dict['sum_x_sq'] += torch.sum(data**2)

def finalize_stats(stats_dict):
    """
    Calculates the final mean and std from the running totals.
    Note: Kurtosis cannot be accurately calculated this way without all data.
    """
    n = stats_dict.get('n', 0)
    if n == 0:
        return None
    
    sum_x = stats_dict.get('sum_x', 0.0)
    sum_x_sq = stats_dict.get('sum_x_sq', 0.0)
    
    mean = sum_x / n
    mean_sq = sum_x_sq / n
    variance = mean_sq - (mean**2)
    # Handle potential floating point inaccuracies where variance is a tiny negative number
    if variance < 0: variance = 0
    std = torch.sqrt(torch.tensor(variance))
    
    return {'mean': mean.item(), 'std': std.item(), 'n': n}

# ==============================================================================
# Main Streaming Analysis Function
# ==============================================================================

def run_streaming_analysis(model_names, category_names=None):
    """
    Run memory-efficient analysis for multiple models using a streaming algorithm.
    """
    if category_names is None:
        category_names = MODEL_TAXONOMY.keys()

    # Initialize data structures for running totals
    streaming_stats = {
        cat: {
            'weights': defaultdict(float),
            'biases': defaultdict(float)
        } for cat in category_names
    }
    
    last_model_categorised_weights = None
    last_model_name = ""

    for model_name in tqdm(model_names, desc="Analyzing models"):
        # Load and classify one model at a time
        categorised_weights = classify_model_parameters(model_name)
        
        for category in category_names:
            if not categorised_weights.get(category):
                continue

            layer_components = categorised_weights[category]
            
            # Extract tensors for the current category
            weight_tensors = [comp['weight'] for comp in layer_components if 'weight' in comp]
            bias_tensors = [comp['bias'] for comp in layer_components if 'bias' in comp]

            # Update the running totals for this category
            update_streaming_stats(streaming_stats[category]['weights'], weight_tensors)
            update_streaming_stats(streaming_stats[category]['biases'], bias_tensors)

        # Keep the classified weights of the last model for plotting a representative sample
        last_model_categorised_weights = categorised_weights
        last_model_name = model_name
        # The large 'categorised_weights' and the model itself are now out of scope
        # and will be garbage-collected in the next loop iteration.

    # --- Finalization and Plotting ---
    print("\n--- Global Statistics (Calculated from all models) ---")
    for category in tqdm(category_names, desc="Finalizing stats and plotting"):
        if not last_model_categorised_weights or not last_model_categorised_weights.get(category):
            continue

        # Finalize the global stats from the running totals
        global_weights_stats = finalize_stats(streaming_stats[category]['weights'])
        global_biases_stats = finalize_stats(streaming_stats[category]['biases'])

        # Get the actual data from the LAST model to plot a representative histogram
        last_model_weights_data, last_model_biases_data = process_params_in_category(last_model_categorised_weights, category)

        # Overwrite the sample stats with the accurate global stats
        if global_weights_stats and last_model_weights_data:
            last_model_weights_data['mean'] = global_weights_stats['mean']
            last_model_weights_data['std'] = global_weights_stats['std']
        
        if global_biases_stats and last_model_biases_data:
            last_model_biases_data['mean'] = global_biases_stats['mean']
            last_model_biases_data['std'] = global_biases_stats['std']

        # Plot the sample distribution annotated with the global stats
        save_path = f"histograms/comparison/{category}_all_models_summary.pdf"
        plot_category_histograms(
            model_name=f"All Models (Sample: {last_model_name})",
            category_name=category,
            weights_data=last_model_weights_data,
            biases_data=last_model_biases_data,
            save_path=save_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory-efficient analysis on Hugging Face models.")
    parser.add_argument("--model_list_file", type=str, default="src/msc_project/analysis/model_names.txt", help="Path to a file containing a list of model names, one per line.")
    args = parser.parse_args()

    if not os.path.exists(args.model_list_file):
        print(f"Error: Model list file not found at {args.model_list_file}")
    else:
        with open(args.model_list_file, "r") as f:
            model_names = [line.strip() for line in f.readlines() if line.strip()]
        
        if not model_names:
            print("Model list file is empty.")
        else:
            run_streaming_analysis(model_names)