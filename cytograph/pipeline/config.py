import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union, Any, Dict

import yaml
import sys
from ..utils import available_cpu_count
import logging
from .punchcards import Punchcard


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
				a[key] = b[key]
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


class Config:
	_singleton: Optional[Dict[str, Any]] = None
	
	@staticmethod
	def load(punchcard: Punchcard = None) -> Dict:
		if Config._singleton is None:
			config: Dict[str, Any] = {}

			# Load the default settings
			load_and_merge_config(config, Path(__file__).resolve().parent / "default_config.yaml")
			# Home directory
			load_and_merge_config(config, Path.home() / ".cytograph")
			# Load config from current folder
			load_and_merge_config(config, Path.cwd() / "config.yaml")

		else:
			config = Config._singleton

		# Load config from current punchcard
		if punchcard is not None:
			merge_config(config, {"punchcards": punchcard.sources, "onlyif": punchcard.onlyif})
			if punchcard.resources is not None:
				merge_config(config, {"resources": punchcard.resources})
			config["punchcard"] = punchcard

		Config._singleton = config
		return config
