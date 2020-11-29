import logging
from typing import Dict
import os
import shoji
from .config import Config
from .punchcards import PunchcardDeck, Punchcard
import cytograph as cg


def run_recipe(ws: shoji.WorkspaceManager, recipe: Dict) -> None:
	for step in recipe:
		for fname, args in step.items():  # This is almost always only a single function and args, but could in principle be several as long as they have distinct names
			logging.info(f"{fname}: {args}")
			# logging.info(f"(not running)")
			instance = getattr(cg, fname)(**args)
			instance.fit(ws, save=True)
			logging.info(f"{fname}: Done.")
			logging.info("")


class Workflow:
	"""
	Run the recipe for a punchcard
	"""
	def __init__(self, deck: PunchcardDeck, punchcard: Punchcard) -> None:
		self.config = Config.load(punchcard)
		self.deck = deck
		self.punchcard = punchcard
		self.export_dir = os.path.join(self.config["paths"]["build"], "exported", punchcard.name)

	def process(self) -> None:
		ws = self.config["workspaces"]["build"][self.punchcard.name]
		logging.info(f"Running recipe for '{self.punchcard.name}'")
		run_recipe(ws, self.config["recipes"][self.punchcard.recipe])
