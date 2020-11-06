import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union, Any, Dict

import yaml

from ..utils import available_cpu_count

from .punchcards import PunchcardSubset, PunchcardView


# https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def merge_config(a, b, path=None):
	"Merge b into a, modifying a in place and returning it"
	if path is None: path = []
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge_config(a[key], b[key], path + [str(key)])
			elif a[key] == b[key]:
				pass  # same leaf value
			else:
				raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
		else:
			a[key] = b[key]
	return a


def load_and_merge_config(config, fname):
	"""
	Load config from fname and merge it into config
	"""
	if os.path.exists(fname):
		with open(fname) as f:
			merge_config(config, yaml.load(f, Loader=yaml.FullLoader))


def load_config(subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = None) -> Dict:
	config: Dict[str, Any] = {}

	# Load the default settings
	load_and_merge_config(config, Path(__file__).resolve().parent / "default_config.yaml")
	# Home directory
	load_and_merge_config(config, Path.home() / ".cytograph")
	# Set build folder
	if config["paths"]["build"] == "":
		config["paths"]["build"] = Path.cwd()
	# Load config from build folder
	load_and_merge_config(config, Path(config["paths"]["build"]) / "config.yaml")
	# Load config from current subset or view
	if subset_obj is not None:
		merge_config(config, {"include": subset_obj.include, "onlyif": subset_obj.onlyif})
		if subset_obj.recipe is not None:
			merge_config(config, {"recipes": {"punchcard": subset_obj.recipe}})
		if subset_obj.execution is not None:
			merge_config(config, {"execution": subset_obj.execution})

	if config["paths"]["workspace"] is None or config["paths"]["workspace"] == "":
		raise ValueError("config.paths.workspace must not be empty")
	return config
