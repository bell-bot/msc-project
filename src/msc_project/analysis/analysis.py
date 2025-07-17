from msc_project.analysis.analysis_utils import (
    classify_model_parameters,
    plot_category_histograms,
    process_params_in_category,
)
from tqdm import tqdm

from msc_project.analysis.constants import MODEL_TAXONOMY


def run_analysis_for_model(model_name, category_names=None):
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

def run_analysis(model_names, category_names=None):
    """
    Run analysis for multiple models.
    :param model_names: List of model names to analyze.
    :param category_names: List of categories to analyze. If None, all categories will be analyzed.
    """

    if category_names is None:
            category_names = MODEL_TAXONOMY.keys()

    weights_data = {key: [] for key in category_names}
    biases_data = {key: [] for key in category_names}

    for model_name in tqdm(model_names, desc="Analyzing models"):
        categorised_weights = classify_model_parameters(model_name)
        
        for category in categorised_weights.keys():
            weights, biases = process_params_in_category(categorised_weights, category)

            if weights or biases:
                # Save histogram for this model and category only
                save_path = f"histograms/{model_name}/{model_name}_{category}.pdf"
                plot_category_histograms(
                    model_name=model_name,
                    category_name=category,
                    weights_data=weights,
                    biases_data=biases,
                    save_path=save_path
                )
                
            if weights:
                weights_data[category].append(weights)
            if biases:
                biases_data[category].append(biases)

    for category in tqdm(category_names, desc="Plotting categories"):
        save_path = f"histograms/comparison/{category}_all.pdf"
        plot_category_histograms(
            model_name=model_names,
            category_name=category,
            weights_data=weights_data[category],
            biases_data=biases_data[category],
            save_path=save_path
        )

if __name__ == "__main__":
    f = open("src/msc_project/analysis/model_names.txt", "r")
    model_names = [line.strip() for line in f.readlines()]
    f.close()
    run_analysis(model_names)
