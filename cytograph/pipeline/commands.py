import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
import click
import numpy as np
import shoji
from .._version import __version__ as version
from .config import Config
from .engine import CondorEngine, Engine, LocalEngine
from .punchcards import PunchcardDeck
from .workflow import Workflow, run_recipe


def pp(config: Dict, indent: int = 0) -> str:
	result = ""
	for k, v in config.items():
		result += " " * indent
		result += k + ": "
		if isinstance(v, dict):
			result += "\n" + pp(v, indent + 2)
		else:
			result += str(v) + "\n"
	return result


@click.group()
@click.option('--show-message/--hide-message', default=True)
@click.option('--verbosity', default="info", type=click.Choice(['error', 'warning', 'info', 'debug']))
def cli(show_message: bool = True, verbosity: str = "info") -> None:
	config = Config.load()
	level = {"error": 40, "warning": 30, "info": 20, "debug": 10}[verbosity]
	logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level, force=True)
	logging.captureWarnings(True)

	if show_message:
		print(f"Cytograph v{version} by Linnarsson Lab 🌸 (http://linnarssonlab.org)")
		print()


@cli.command()
@click.option('--engine', default="local", type=click.Choice(['local', 'condor']))
@click.option('--dryrun/--no-dryrun', is_flag=True, default=False)
def build(engine: str, dryrun: bool) -> None:
	try:
		config = Config.load()
		Path(config['paths']['build']).mkdir(exist_ok=True)

		# Load the punchcard deck
		deck = PunchcardDeck(config['paths']['build'])

		# Create the execution engine
		execution_engine: Optional[Engine] = None
		if engine == "local":
			execution_engine = LocalEngine(deck, dryrun)
		elif engine == "condor":
			execution_engine = CondorEngine(deck, dryrun)

		# Execute the build
		assert(execution_engine is not None)
		execution_engine.execute()
	except Exception as e:
		logging.exception(f"'build' command failed: {e}")


@cli.command()
@click.argument("punchcard")
@click.option('--resume', default=0)
def process(punchcard: str, resume: int) -> None:
	workspace = Path(os.getcwd()).name
	logging.info(f"Using '{workspace}' as the workspace")
	try:
		config = Config.load()  # This config will not have subset-specific settings, but we need it for the build path
		if not Path(config['paths']['builds']) in Path(os.getcwd()).parents:
			logging.error(f"Current folder '{os.getcwd()}' is not a subfolder of the configured build folder '{config['paths']['builds']}' ")
			sys.exit(1)
		config["paths"]["build"] = Path(config['paths']['builds']) / workspace
		logging.info(f"Build folder is '{config['paths']['build']}'")
		config["paths"]["build"].mkdir(exist_ok=True)

		db = shoji.connect()
		if config['workspaces']['builds'] not in db:
			db[config['workspaces']['builds']] = shoji.Workspace()
		ws_builds = db[config['workspaces']['builds']]
		if workspace not in ws_builds:
			logging.info(f"Creating Workspace '{config['workspaces']['builds']}.{workspace}'")
			ws_builds[workspace] = shoji.Workspace()
		
		if resume > 0:
			if punchcard not in ws_builds[workspace]:
				logging.error(f"Cannot resume, because workspace '{config['workspaces']['builds']}.{workspace}.{punchcard}' does not exist")
				sys.exit(1)
		else:
			if punchcard in ws_builds[workspace]:
				logging.warning(f"Deleting existing Workspace '{config['workspaces']['builds']}.{workspace}.{punchcard}'")
				del ws_builds[workspace][punchcard]
			logging.info(f"Creating Workspace '{config['workspaces']['builds']}.{workspace}.{punchcard}'")
			ws_builds[workspace][punchcard] = shoji.Workspace()
		config["workspaces"]["build"] = ws_builds[workspace]

		deck = PunchcardDeck(config['paths']['build'] / "punchcards")
		punchcard_obj = deck.punchcards[punchcard]
		if punchcard_obj is None:
			logging.error(f"Punchcard {punchcard} not found.")
			sys.exit(1)

		config = Config.load(punchcard_obj)  # Ensure we get any punchcard-specific configs
		for line in pp(config).split("\n"):
			logging.debug(line)

		logging.info(f"Processing '{punchcard}'")
		Workflow(deck, punchcard_obj).process(resume)
	except Exception as e:
		logging.exception(f"'process' command failed: {e}")
		sys.exit(1)

@cli.command()
@click.argument('sampleids', nargs=-1)
@click.option("--force", default=False, is_flag=True)
def qc(sampleids: List[str], force: bool) -> None:
	try:
		workspace = Path(os.getcwd()).name
		logging.info(f"Using '{workspace}' as the workspace")

		config = Config.load()
		for line in pp(config).split("\n"):
			logging.debug(line)
		config["paths"]["build"] = Path(config['paths']['builds']) / workspace
		logging.info(f"Build folder is '{config['paths']['build']}'")
		config["paths"]["build"].mkdir(exist_ok=True)

		deck = PunchcardDeck(config['paths']['build'] / "punchcards")

		sampleids = np.unique(sampleids)
		db = shoji.connect()
		for sampleid in sampleids:
			ws = db[config["workspaces"]["samples"]]
			if sampleid in ws:
				if force or "PassedQC" not in ws[sampleid]:
					logging.info(f"Processing '{sampleid}'")
					recipe = config["recipes"]["qc"]
					run_recipe(ws[sampleid], recipe)
				else:
					logging.info(f"Skipping '{sampleid}' because QC already done (use --force to override)")
			elif sampleid in deck.punchcards:
				punchcard = deck.punchcards[sampleid]
				for sample in punchcard.sources:
					if sample in ws:
						if force or "PassedQC" not in ws[sample]:
							logging.info(f"Processing '{sample}'")
							recipe = config["recipes"]["qc"]
							run_recipe(ws[sample], recipe)
						else:
							logging.info(f"Skipping '{sample}' because QC already done (use --force to override)")
					else:
						logging.warning(f"Skipping '{sample}' (specificed in punchcard '{sampleid}') because sample not found")
	except Exception as e:
		logging.error(f"'qc' command failed: {e}")
		sys.exit(1)