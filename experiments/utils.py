from datetime import datetime

def generate_experiment_id(experiment_type):
    
    experiment_id = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return experiment_id

def save_experiment_info(experiment_id, info, save_path):
    """
    Save experiment information to a file.
    
    Args:
        experiment_id (str): Unique identifier for the experiment.
        info (dict): Information about the experiment.
        save_path (str): Path to save the experiment information file.
    """
    with open(f"{save_path}/{experiment_id}_info.txt", "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")