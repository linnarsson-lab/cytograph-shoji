import logging
from typing import Dict
import numpy as np
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


def run_qc(ws: shoji.WorkspaceManager, recipe: Dict) -> None:
	start_all = datetime.now()
	config = Config.load()
	export_dir = config.path / "exported" / "qc"
	export_dir.mkdir(exist_ok=True)
	
	for step in recipe:
		for fname, args in step.items():  # This is almost always only a single function and args, but could in principle be several as long as they have distinct names
			logging.info(f"{fname}: {args}")
			start = datetime.now()
			# logging.info(f"(not running)")
			instance = getattr(cg, fname)(**args)
			instance.export_dir = export_dir
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
		self.export_dir = self.config.path / "exported" / punchcard.name
		self.export_dir.mkdir(exist_ok=True)

	def process(self, resume_at: int = 0) -> None:
		logdir: Path = self.config.path / "logs"
		logdir.mkdir(exist_ok=True)
		assert self.config.workspaces.build is not None
		ws = self.config.workspaces.build[self.punchcard.name]
		if resume_at == 0:
			logging.info(f"Running recipe '{self.punchcard.recipe}' for '{self.punchcard.name}'")
		else:
			logging.info(f"Resuming recipe '{self.punchcard.recipe}' from step {resume_at} for '{self.punchcard.name}'")
		recipe = self.config.recipes[self.punchcard.recipe][resume_at:]
		steps = np.array("\n".join([str(s) for s in recipe]), dtype=object)
		if "Recipe" not in ws:
			ws.Recipe = shoji.Tensor(dtype="string", dims=(), inits=steps)
		else:
			ws.Recipe.append({"Recipe": steps})
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
