from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import shoji

from .punchcards import Punchcard


@dataclass
class ResourceConfig:
	n_cpus: int
	n_gpus: int
	memory: int


@dataclass
class WorkspacesConfig:
	samples_workspace_name: str
	builds_root_workspace_name: str
	build: Optional[shoji.WorkspaceManager]


class Config:
	_singleton = None

	def __init__(self) -> None:
		self.workspaces = WorkspacesConfig("", "", None)
		self.resources = ResourceConfig(0, 0, 0)
		self.recipes: Dict[str, Any] = {}
		self.sources: List[str] = []
		self.onlyif: str = ""
		self.punchcard: Optional[Punchcard] = None
		self.path: Path = Path.cwd()

	def __repr__(self) -> str:
		s = f"workspaces:\n  builds_root: {self.workspaces.builds_root_workspace_name}\n  samples: {self.workspaces.samples_workspace_name}\n"
		s += f"resources:\n  n_cpus: {self.resources.n_cpus}\n  n_gpus: {self.resources.n_gpus}\n  memory: {self.resources.memory}\n"
		s += "recipes:\n  " + "\n  ".join(self.recipes.keys()) + "\n"
		s += "sources: [" + ",".join(self.sources) + "]\n"
		s += f'onlyif: "{self.onlyif}"'
		return s

	def _merge(self, cfg: Dict[str, Any]) -> None:
		if "workspaces" in cfg:
			assert "builds_root" in cfg["workspaces"], "'workspaces' config must contain key 'builds_root' indicating the root workspace for builds in Shoji"
			self.workspaces.builds_root_workspace_name = cfg["workspaces"]["builds_root"]
			assert "samples" in cfg["workspaces"], "'workspaces' config must contain key 'samples' indicating the workspace for samples in Shoji"
			self.workspaces.samples_workspace_name = cfg["workspaces"]["samples"]
			# Note the build workspace cannot be configured; it's inferred from the working directory
		if "resources" in cfg:
			assert "n_cpus" in cfg["resources"], "'resources' config must contain key 'n_cpus' indicating the number of CPUs to request"
			assert "n_gpus" in cfg["resources"], "'resources' config must contain key 'n_gpus' indicating the number of GPUs to request"
			assert "memory" in cfg["resources"], "'resources' config must contain key 'memory' indicating the amount (GB) of memory to request"
			self.resources.n_cpus = int(cfg["resources"]["n_cpus"])
			self.resources.n_gpus = int(cfg["resources"]["n_gpus"])
			self.resources.memory = int(cfg["resources"]["memory"])
		if "recipes" in cfg:
			for name, recipe in cfg["recipes"].items():
				self.recipes[name] = recipe
		if "sources" in cfg:
			self.sources = cfg["sources"]
		if "onlyif" in cfg:
			self.onlyif = cfg["onlyif"]

	def _load_and_merge(self, fname: Path) -> None:
		with open(fname) as f:
			self._merge(yaml.load(f, Loader=yaml.FullLoader))

	@staticmethod
	def load(punchcard: Punchcard = None) -> "Config":
		if Config._singleton is None:
			config = Config()
			# Load the default settings
			config._load_and_merge(Path(__file__).resolve().parent / "default_config.yaml")
			# Home directory
			config._load_and_merge(Path.home() / ".cytograph")
			# Load config from current folder
			config._load_and_merge(Path.cwd() / "config.yaml")
		else:
			config = Config._singleton

		# Load config from current punchcard
		if punchcard is not None:
			config._merge({
				"punchcards": punchcard.sources,
				"onlyif": punchcard.onlyif
			})
			if punchcard.resources is not None:
				config._merge({"resources": punchcard.resources})
			config.punchcard = punchcard

		Config._singleton = config
		return config
