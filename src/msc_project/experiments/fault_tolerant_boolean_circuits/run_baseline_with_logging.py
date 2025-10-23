import numpy as np

from msc_project.experiments.fault_tolerant_boolean_circuits.experiment import run_baseline_with_logging
from msc_project.utils.experiment_utils import RobustnessExperimentSpecs

noise_stds = np.linspace(0.0010, 0.02, 20)

experiment_config = RobustnessExperimentSpecs(
    experiment_name="baseline_majority_vote_15",
    backdoor_type="baseline_majority_vote",
    noise_stds=noise_stds,
    num_samples=50,
    c=20,
    n=3,
    log_w=1,
    redundancy=15,
)
print(experiment_config)

run_baseline_with_logging(experiment_config)
