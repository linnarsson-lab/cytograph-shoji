import logging
import os
import sys
from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path
from .config import ResourceConfig


class Punchcard:
	def __init__(self, path: Path) -> None:
		items = path.name.split(".")
		if len(items) > 2:
			logging.error("Punchcard name cannot contain dots (apart from .yaml suffix)")
			sys.exit(1)
		self.name = items[0]
		if not path.exists():
			logging.error(f"Punchcard '{path}' not found.")
			sys.exit(1)
		with open(path) as f:
			spec: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)
		self.onlyif = spec.get("onlyif", None)
		sources_spec = spec.get("sources", [])
		self.sources: List[str] = []
		self.sources_onlyif: List[Optional[str]] = []
		for s in sources_spec:
			if ";" in s:
				self.sources.append(s.split(";")[0])
				self.sources_onlyif.append(s.split(";")[1])
			else:
				self.sources.append(s)
				self.sources_onlyif.append(self.onlyif)
		self.recipe = spec.get("recipe", "punchcard")
		resources = spec.get("resources", {})
		n_cpus = resources.get("n_cpus", 1)
		n_gpus = resources.get("n_gpus", 0)
		memory = resources.get("memory", 8)
		self.resources = ResourceConfig(n_cpus, n_gpus, memory)

	@staticmethod
	def load_all(path: Path) -> List["Punchcard"]:
		result: List[Punchcard] = []
		if path.exists():
			for f in os.listdir(path):
				if f.lower().endswith(".yaml"):
					result.append(Punchcard(path / f))
		return result


class PunchcardDeck:
	def __init__(self, punchcard_dir: Path) -> None:
		self.punchcards: Dict[str, Punchcard] = {}
		for p in Punchcard.load_all(punchcard_dir):
			self.punchcards[p.name] = p
