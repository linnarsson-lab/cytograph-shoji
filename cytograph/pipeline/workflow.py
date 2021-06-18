import logging
from typing import Dict
import os
import shoji
from .config import Config
from .punchcards import PunchcardDeck, Punchcard
import cytograph as cg
from datetime import datetime
from pathlib import Path


def nice_deltastring(delta):
	result = []
	s = delta.total_seconds()
	h = s // 60 // 60
	if h >= 1:
		result.append(f"{int(h)}h")
		s -= h * 60 * 60
	m = s // 60
	if m >= 1:
		result.append(f"{int(m)}m")
		s -= m * 60
	if s >= 1:
		result.append(f"{int(s)}s")
	if len(result) > 0:
		return " ".join(result)
	else:
		return f"{delta.microseconds // 1000} ms"


def run_recipe(ws: shoji.WorkspaceManager, recipe: Dict) -> None:
	start_all = datetime.now()
	for step in recipe:
		for fname, args in step.items():  # This is almost always only a single function and args, but could in principle be several as long as they have distinct names
			logging.info(f"{fname}: {args}")
			start = datetime.now()
			# logging.info(f"(not running)")
			instance = getattr(cg, fname)(**args)
			instance.fit(ws, save=True)
			end = datetime.now()
			logging.info(f"{fname}: Done in {nice_deltastring(end - start)}.")
			logging.info("")
	end_all = datetime.now()
	logging.info(f"Recipe completed in {nice_deltastring(end_all - start_all)}.")
	logging.info("")


class Workflow:
	"""
	Run the recipe for a punchcard
	"""
	def __init__(self, deck: PunchcardDeck, punchcard: Punchcard) -> None:
		self.config = Config.load(punchcard)
		self.deck = deck
		self.punchcard = punchcard
		self.export_dir = Path(os.path.join(self.config["paths"]["build"], "exported", punchcard.name))
		os.makedirs(self.export_dir, exist_ok=True)

	def process(self, resume_at: int = 0) -> None:
		logdir: Path = self.config["paths"]["build"] / "logs"
		logdir.mkdir(exist_ok=True)
		ws = self.config["workspaces"]["build"][self.punchcard.name]
		logging.info(f"Running recipe '{self.punchcard.recipe}' for '{self.punchcard.name}'")
		recipe = self.config["recipes"][self.punchcard.recipe][resume_at:]
		start_all = datetime.now()
		for step in recipe:
			for fname, args in step.items():
				logging.info(f"{fname}: {args}")
				start = datetime.now()
				instance = getattr(cg, fname)(**args)
				instance.export_dir = self.export_dir
				instance.fit(ws, save=True)
				end = datetime.now()
				logging.info(f"{fname}: Done in {nice_deltastring(end - start)}.")
				logging.info("")
		end_all = datetime.now()
		logging.info(f"Recipe completed in {nice_deltastring(end_all - start_all)}.")
		logging.info("")

		(logdir / (self.punchcard.name + ".completed")).touch()
		(logdir / (self.punchcard.name + ".created")).unlink(missing_ok=True)
