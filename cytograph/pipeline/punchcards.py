import logging
import os
import sys
from typing import Any, Dict, List, Optional
import yaml


class Punchcard:
	def __init__(self, path: str) -> None:
		items = os.path.basename(path).split(".")
		if len(items) > 2:
			logging.error("Punchcard name cannot contain dots (apart from .yaml suffix)")
			sys.exit(1)
		self.name = items[0]
		if not os.path.exists(path):
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
		self.resources = spec.get("resources")

	@staticmethod
	def load_all(path: str) -> List["Punchcard"]:
		result: List[Punchcard] = []
		if os.path.exists(path):
			for f in os.listdir(path):
				if f.lower().endswith(".yaml"):
					result.append(Punchcard(os.path.join(path, f)))
		return result


class PunchcardDeck:
	def __init__(self, punchcard_dir: str) -> None:
		self.punchcards: Dict[str, Punchcard] = {}
		for p in Punchcard.load_all(punchcard_dir):
			self.punchcards[p.name] = p
