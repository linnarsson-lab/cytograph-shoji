import logging
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Set
import click
import numpy as np
import shoji
from .._version import __version__ as version
from .config import Config
from .engine import CondorDAGEngine, Engine, LocalEngine, CondorEngine
from .punchcards import PunchcardDeck
from .workflow import Workflow, run_qc


@click.group()
@click.option('--show-message/--hide-message', default=True)
@click.option('--verbosity', default="info", type=click.Choice(['error', 'warning', 'info', 'debug']))
def cli(show_message: bool = True, verbosity: str = "info") -> None:
	level = {"error": 40, "warning": 30, "info": 20, "debug": 10}[verbosity]
	if sys.version_info >= (3, 8):
		logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level, force=True)
	else:
		logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
	logging.captureWarnings(True)

	if show_message:
		print(f"Cytograph v{version} by Linnarsson Lab 🌸 (http://linnarssonlab.org)")
		print()


@cli.command()
@click.option('--engine', type=click.Choice(['local', 'condor_dag', 'condor']))
@click.option('--dryrun/--no-dryrun', is_flag=True, default=False)
def build(engine: str, dryrun: bool) -> None:
	try:
		config = Config.load()
		if not (Path.cwd() / "punchcards").exists():
			logging.info("Current folder is not a proper build folder (no 'punchcards' sub-folder)")
			sys.exit(1)
		logging.info(f"Build folder is '{config.path}'")

		# Load the punchcard deck
		deck = PunchcardDeck(config.path / "punchcards")

		# Create the execution engine
		execution_engine: Optional[Engine] = None
		if engine == "local":
			execution_engine = LocalEngine(deck, dryrun)
		elif engine == "condor_dag":
			execution_engine = CondorDAGEngine(deck, dryrun)
		elif engine == "condor":
			execution_engine = CondorEngine(deck, dryrun)
		else:
			logging.error("No engine was specified")
			sys.exit(1)

		# Execute the build
		assert(execution_engine is not None)
		execution_engine.execute()
	except Exception as e:
		logging.exception(f"'build' command failed: {e}")


@cli.command()
def init() -> None:
	"""
	Initialize a build folder with default config and sub-folders.
	"""
	try:
		logging.info("Initializing build folder")
		cwd = Path().cwd()
		parent = cwd.parent
		if Path('/proj/cytograph') not in parent.parents:
			logging.error(f"WARNING: build folder {parent} is not a subdirectory of '/proj/cytograph'")
			if not query_yes_no("Do you want to proceed anyway?"):
				sys.exit(1)

		(cwd / "punchcards").mkdir(exist_ok=True)
		(cwd / "exported").mkdir(exist_ok=True)
		(cwd / "logs").mkdir(exist_ok=True)
		shutil.copyfile(Path(__file__).resolve().parent / "default_config.yaml", cwd / "config.yaml")
		with open(Path(__file__).resolve().parent / "default_config.yaml") as fin:
			with open(cwd / "config.yaml", "w") as fout:
				fout.write(fin.read().replace("$$BUILDNAME$$", parent.name))
		logging.info(f"Created default config file and folders")

		db = shoji.connect()
		config = Config.load()
		build_root_parts = config.workspaces.builds_root_workspace_name.split(".")
		ws = db
		for part in build_root_parts:
			if part not in ws:
				ws[part] = shoji.Workspace()
			ws = ws[part]
		logging.info(f"Created build workspace '{config.workspaces.builds_root_workspace_name}' in Shoji")

		shutil.copyfile(Path(__file__).resolve().parent / "default_punchcard.yaml", cwd / "punchcards" / "Example.yaml")
		logging.info(f"Created example punchcard")

		logging.info(f"Done; build folder is '{config.path}'.")
	except Exception as e:
		logging.exception(f"'build' command failed: {e}")


@cli.command()
@click.argument("punchcard")
@click.option('--resume', default=0)
@click.option('--recipe')
def process(punchcard: str, resume: int, recipe: str) -> None:
	workspace_name = Path.cwd().name
	logging.info(f"Workspace is '{workspace_name}'")
	try:
		config = Config.load()
		if not (Path.cwd() / "punchcards").exists():
			logging.info("Current folder is not a proper build folder (no 'punchcards' sub-folder)")
			sys.exit(1)
		logging.info(f"Build folder is '{config.path}'")

		db = shoji.connect()
		if config.workspaces.builds_root_workspace_name not in db:
			db[config.workspaces.builds_root_workspace_name] = shoji.Workspace()
		builds_root_ws = db[config.workspaces.builds_root_workspace_name]
		if workspace_name not in builds_root_ws:
			logging.info(f"Creating Workspace '{config.workspaces.builds_root_workspace_name}.{workspace_name}'")
			builds_root_ws[workspace_name] = shoji.Workspace()
		
		build_ws = builds_root_ws[workspace_name]
		config.workspaces.build = build_ws
		if resume > 0:
			if punchcard not in build_ws:
				logging.error(f"Cannot resume, because workspace '{config.workspaces.builds_root_workspace_name}.{workspace_name}.{punchcard}' does not exist")
				sys.exit(1)
		else:
			if punchcard in build_ws:
				logging.warning(f"Deleting existing Workspace '{config.workspaces.builds_root_workspace_name}.{workspace_name}.{punchcard}'")
				del build_ws[punchcard]
			logging.info(f"Creating Workspace '{config.workspaces.builds_root_workspace_name}.{workspace_name}.{punchcard}'")
			build_ws[punchcard] = shoji.Workspace()

		deck = PunchcardDeck(config.path / "punchcards")
		punchcard_obj = deck.punchcards[punchcard]
		if punchcard_obj is None:
			logging.error(f"Punchcard {punchcard} not found.")
			sys.exit(1)

		config = Config.load(punchcard_obj)  # Ensure we get any punchcard-specific configs
		if recipe is not None:
			punchcard_obj.recipe = recipe
		logging.info(f"Processing '{punchcard}'")
		Workflow(deck, punchcard_obj).process(resume)
	except Exception as e:
		logging.exception(f"'process' command failed: {e}")
		sys.exit(1)

@cli.command()
@click.argument('sampleids', nargs=-1)
@click.option("--force", default=False, is_flag=True)
@click.option("--import-from", default="")
def qc(sampleids: List[str], force: bool, import_from: str) -> None:
	workspace_name = Path.cwd().name
	logging.info(f"Workspace is '{workspace_name}'")
	try:
		config = Config.load()
		if not (Path.cwd() / "punchcards").exists():
			logging.info("Current folder is not a proper build folder (no 'punchcards' sub-folder)")
			sys.exit(1)
		logging.info(f"Build folder is '{config.path}'")

		deck = PunchcardDeck(config.path / "punchcards")
		if sampleids[0] in deck.punchcards:
			punchcard = deck.punchcards[sampleids[0]]
			sampleids = punchcard.sources
		sampleids = np.unique(sampleids)
		db = shoji.connect()
		for sampleid in sampleids:
			if sampleid.startswith("10X"):
				logging.warning("Sample names starting with numbers are discouraged and may cause errors; e.g. use TenX101_1 instead of 10X101_1")
			ws = db[config.workspaces.samples_workspace_name]
			if import_from != "" and import_from is not None:
				if force or sampleid not in ws:
					del ws[sampleid]
					ws[sampleid] = shoji.Workspace()
					logging.info(f"Importing '{sampleid}' from '{import_from}'")
					ws[sampleid]._from_loom(Path(import_from) / (sampleid.replace("TenX", "10X") + ".loom"))
			if sampleid not in ws:
				logging.info(f"Skipping '{sampleid}' because sample missing (use --import-from <path> to auto-import)")
				continue
			if force or "PassedQC" not in ws[sampleid]:
				logging.info(f"Processing '{sampleid}'")
				recipe = config.recipes["qc"]
				run_qc(ws[sampleid], recipe)
			else:
				logging.info(f"Skipping '{sampleid}' because QC already done (use --force to override)")
	except Exception as e:
		logging.error(f"'qc' command failed: {e}")
		sys.exit(1)


@cli.command()
@click.argument('workspace')
@click.argument('loomfiles', nargs=-1)
def import_(workspace: str, loomfiles: List[str]) -> None:
	try:
		db = shoji.connect()
		ws = db[workspace]
		for lf in loomfiles:
			logging.info(f"Importing '{lf}'")
			ws._from_loom(lf)
	except Exception as e:
		logging.error(f"'import' command failed: {e}")
		sys.exit(1)


@cli.command()
def leaves() -> None:
	try:
		config = Config.load()
		if not (Path.cwd() / "punchcards").exists():
			logging.info("Current folder is not a proper build folder (no 'punchcards' sub-folder)")
			sys.exit(1)

		deck = PunchcardDeck(config.path / "punchcards")
		dag = Engine(deck).build_execution_dag()
		parents: Set[str] = set()
		for dependencies in dag.values():
			for p in dependencies:
				parents.add(p)
		leaves = [p for p in dag.keys() if p not in parents]

		print(", ".join(leaves))

	except Exception as e:
		logging.error(f"'leaves' command failed: {e}")
		sys.exit(1)
