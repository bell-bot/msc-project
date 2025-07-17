from msc_project.analysis.analysis_utils import (
    classify_model_parameters,
    plot_category_histograms,
    process_params_in_category,
)
from tqdm import tqdm


def run_analysis(model_name, category_names=None):
    """
    Run complete analysis on the model weights and plot the distribution for all categories.
    If category_names is provided, only those categories will be analyzed.
    """

    categorised_weights = classify_model_parameters(model_name)
    if category_names is None:
        category_names = categorised_weights.keys()
        for category in tqdm(category_names, desc="Analyzing categories"):
            if categorised_weights[category]:
                weights_data, biases_data = process_params_in_category(categorised_weights, category)
                plot_category_histograms(
                    weights_data, biases_data, category, model_name, save_path=None
                )
    elif isinstance(category_names, str):
        category_names = [category_names]
        for category in tqdm(category_names, desc="Analyzing categories"):
            if categorised_weights[category]:
                weights_data, biases_data = process_params_in_category(categorised_weights, category)
                plot_category_histograms(
                    weights_data, biases_data, category, model_name, save_path=None
                )
    elif isinstance(category_names, list):
        for category in tqdm(category_names, desc="Analyzing categories"):
            if categorised_weights[category]:
                weights_data, biases_data = process_params_in_category(categorised_weights, category)
                plot_category_histograms(
                    model_name=model_name,
                    category_name=category,
                    weights_data=weights_data,
                    biases_data=biases_data,
                    save_path=None
                )


if __name__ == "__main__":
    run_analysis("distilbert/distilgpt2", category_names=["attention_query", "mlp_up", "final_norm"])
