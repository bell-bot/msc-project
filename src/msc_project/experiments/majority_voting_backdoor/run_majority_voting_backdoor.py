import numpy as np

from msc_project.circuits_custom.custom_stepmlp import MajorityVotingStepMLP
from msc_project.utils.experiment_utils import RobustnessExperimentSpecs
from msc_project.experiments.majority_voting_backdoor.experiment import run

noise_stds = np.linspace(0.0010, 0.02, 200)

experiment_config = RobustnessExperimentSpecs(
    experiment_name="noise_robustness_redundancy_4_small",
    noise_stds=noise_stds,
    redundancy=4,
    num_samples=50,
    c=20,
    n=3,
    log_w=1
)

run(experiment_config)