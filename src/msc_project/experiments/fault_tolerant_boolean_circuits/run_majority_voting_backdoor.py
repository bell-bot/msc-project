import numpy as np

from msc_project.experiments.fault_tolerant_boolean_circuits.experiment import run_majority_voting
from msc_project.utils.experiment_utils import RobustnessExperimentSpecs

noise_stds = np.linspace(0.0010, 0.02, 200)

experiment_config = RobustnessExperimentSpecs(
    experiment_name="majority_voting_2_small",
    noise_stds=noise_stds,
    redundancy=4,
    num_samples=50,
    c=20,
    n=3,
    log_w=1
)

run_majority_voting(experiment_config)