import logging
import os
import shoji
from .config import load_config
from .punchcards import PunchcardDeck
from .utils import run_recipe


class Workflow:
	"""
	Run the recipe for a punchcard
	"""
	def __init__(self, deck: PunchcardDeck, name: str) -> None:
		self.config = load_config()
		self.deck = deck
		self.name = name
		self.export_dir = os.path.join(self.config["paths"]["build"], "exported", name)

	def process(self) -> None:
		db = shoji.connect()
		ws = db[self.config["paths"]["workspace"] + "." + self.name]
		logging.info(f"Running recipe for '{self.name}'")
		run_recipe(ws, self.config["recipes"]["punchcard"])
