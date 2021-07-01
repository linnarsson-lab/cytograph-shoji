from dataclasses import dataclass


@dataclass
class ResourceConfig:
	n_cpus: int
	n_gpus: int
	memory: int
