from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class ExperimentSpecs:

    target_model: str
    experiment_name: str

    num_samples: int = 50
    c: int | None = 448
    n: int = 24
    log_w: Literal[0, 1, 2, 3, 4, 5, 6] = 6
    random_seed: int = 95
    trigger_length: int = 16
    payload_length: int = 16
    sample_size: int = 1000000

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
