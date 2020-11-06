import fnmatch
import logging
import os
import sqlite3 as sqlite
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union
from ..plotting import qc_plots
import click
import numpy as np
import shoji
import cytograph as cg
from .utils import run_recipe
from ..postprocessing import split_subset, merge_subset
from .._version import __version__ as version
from .config import load_config
from .engine import CondorEngine, Engine, LocalEngine
from .punchcards import PunchcardDeck, PunchcardSubset, PunchcardView
from .workflow import Workflow
import subprocess
import shutil
import time
import getpass
import datetime


def make_build_name():
	user = getpass.getuser()
	now = datetime.datetime.now()
	return f"{user}_{now.year}_{now.month}_{now.day}_" + "".join(np.random.choice(["A", "C", "G", "T"], size=6))


@click.group()
@click.option('--build-location')
@click.option('--show-message/--hide-message', default=True)
@click.option('--verbosity', default="info", type=click.Choice(['error', 'warning', 'info', 'debug']))
def cli(build_location: str = None, show_message: bool = True, verbosity: str = "info") -> None:
	config = load_config()
	level = {"error": 40, "warning": 30, "info": 20, "debug": 10}[verbosity]
	logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
	logging.captureWarnings(True)

	# Allow command-line options to override config settings
	if build_location is not None:
		config['paths']['build'] = build_location

	if show_message:
		print(f"Cytograph v{version} by Linnarsson Lab 🌸 (http://linnarssonlab.org)")
		if os.path.exists(config['paths']['build']):
			print(f"            Build: {config['paths']['build']}")
		else:
			print(f"            Build: {config['paths']['build']} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config['paths']['samples']):
			print(f"          Samples: {config['paths']['samples']}")
		else:
			print(f"          Samples: {config['paths']['samples']} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config['paths']['autoannotation']):
			print(f"  Auto-annotation: {config['paths']['autoannotation']}")
		else:
			print(f"  Auto-annotation: {config['paths']['autoannotation']} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config['paths']['metadata']):
			print(f"         Metadata: {config['paths']['metadata']}")
		else:
			print(f"         Metadata: {config['paths']['metadata']} \033[1;31;40m-- FILE DOES NOT EXIST --\033[0m")
		print(f"   Fastq template: {config['paths']['fastqs']}")
		if os.path.exists(config['paths']['index']):
			print(f"            Index: {config['paths']['index']}")
		else:
			print(f"            Index: {config['paths']['index']} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config['paths']['qc']):
			print(f"     QC directory: {config['paths']['qc']}")
		else:
			print(f"     QC directory: {config['paths']['qc']} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		print()


@cli.command()
@click.option('--engine', default="local", type=click.Choice(['local', 'condor']))
@click.option('--dryrun/--no-dryrun', is_flag=True, default=False)
def build(engine: str, dryrun: bool) -> None:
	try:
		config = load_config()
		Path(config['paths']['build']).mkdir(exist_ok=True)
		config["paths"]["workspace"] = make_build_name()

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
@click.argument("subset_or_view")
@click.argument("--workspace", default="")
def process(subset_or_view: str, workspace: str) -> None:
	try:
		config = load_config()  # This config will not have subset-specific settings, but we need it for the build path
		Path(config['paths']['build']).mkdir(exist_ok=True)
		if workspace == "":
			config["paths"]["workspace"] = "builds." + sten_2020_11_01_GCATCA
		else:
			config["paths"]["workspace"] = workspace

		logging.info(f"Processing '{subset_or_view}'")

		deck = PunchcardDeck(config['paths']['build'])
		subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = deck.get_subset(subset_or_view)
		if subset_obj is None:
			subset_obj = deck.get_view(subset_or_view)
			if subset_obj is None:
				logging.error(f"Subset or view {subset_or_view} not found.")
				sys.exit(1)

		Workflow(deck, subset_obj).process()
	except Exception as e:
		logging.exception(f"'process' command failed: {e}")


@cli.command()
@click.argument('sampleids', nargs=-1)
@click.option('--file', help="Path to file containing sample IDs (one ID per line)")
def qc(sampleids: List[str], file: str = None) -> None:
	config = load_config()
	Path(config['paths']['build']).mkdir(exist_ok=True)

	samples_to_process: List[str] = []
	if len(sampleids) > 0:
		samples_to_process = sampleids

	if file is not None:
		with open(file) as f:
			for sample in f:
				if "," in sample:
					logging.error("Sample IDs in file cannot contain comma (,); put one sample ID per line")
				samples_to_process.append(sample)

	sampleids = np.unique(samples_to_process)
	db = shoji.connect()
	for sampleid in sampleids:
		ws = db[config["paths"]["samples"] + "." + sampleid]
		recipe = config["recipes"]["qc"]
		run_recipe(ws, recipe)



@cli.command()
@click.option('--subset', default=None)
@click.option('--method', default='svc', type=click.Choice(['svc', 'dendrogram', 'cluster']))
def split(subset: str = None, method: str = 'svc') -> None:

	config = load_config()

	if subset:

		logging.info(f"Splitting {subset}...")
		if split_subset(config, subset, method):
			deck = PunchcardDeck(config['paths']['build'])
			card = deck.get_card(subset)
			Workflow(deck, "").compute_subsets(card)
			logging.error(f"Done.")
		else:
			logging.error(f"Subset cannot be split further.")

	else:
		logging.info(f"Splitting build...")
		cytograph_exe = shutil.which('cytograph')
		exportdir = os.path.abspath(os.path.join(config['paths']['build'], "exported"))
		exdir = os.path.abspath(os.path.join(config['paths']['build'], "split"))
		if not os.path.exists(exdir):
			os.mkdir(exdir)

		# Run build before starting
		logging.info("Making sure the build is complete before splitting...")
		subprocess.run(["cytograph", "build", "--engine", "condor"])

		# Wait until build has been processed
		deck = PunchcardDeck(config['paths']['build'])
		leaves = deck.get_leaves()
		done = False
		while not done:
			time.sleep(30)
			logging.info('Checking build...')
			done = True
			for subset in leaves:
				f = os.path.join(exportdir, subset.longname())
				if not os.path.exists(f):
					done = False

		split = False
		while not split:

			for subset in leaves:

				# Check if dataset was fit already
				f = os.path.join(exportdir, subset.longname(), method)
				if not os.path.exists(f):

					# get command for task
					task = subset.longname()
					cmd = f"split --subset {task} --method {method}"

					# create submit file for split
					with open(os.path.join(exdir, task + ".condor"), "w") as f:
						f.write(f"""
			getenv       = true
			executable   = {os.path.abspath(cytograph_exe)}
			arguments    = "{cmd}"
			log          = {os.path.join(exdir, task)}.log
			output       = {os.path.join(exdir, task)}.out
			error        = {os.path.join(exdir, task)}.error
			request_cpus = 7
			queue 1\n
			""")

					# Submit
					subprocess.run(["condor_submit", os.path.join(exdir, task + ".condor")])

			logging.info("Splitting leaves")
			# Wait until all leaves have been checked for splitting
			done = False
			while not done:
				time.sleep(30)
				logging.info('Checking for split...')
				done = True
				for subset in leaves:
					f = os.path.join(exportdir, subset.longname(), method)
					if not os.path.exists(f):
						done = False

			# Run build
			logging.info("Processing new build")
			subprocess.run(["cytograph", "build", "--engine", "condor"])

			if method != 'svc':
				return

			# Wait until all new subsets have been processed
			deck = PunchcardDeck(config['paths']['build'])
			leaves = deck.get_leaves()
			done = False
			while not done:
				time.sleep(30)
				logging.info('Checking build...')
				done = True
				for subset in leaves:
					f = os.path.join(exportdir, subset.longname())
					if not os.path.exists(f):
						done = False

			# Check if all leaves have been checked for splitting
			logging.info("Checking if all leaves have been split...")
			split = True
			for subset in leaves:
				f = os.path.join(exportdir, subset.longname(), method)
				if not os.path.exists(f):
					split = False


@cli.command()
@click.option('--subset', default=None)
@click.option('--overwrite', is_flag=True)
def merge(subset: str = None, overwrite: bool = False) -> None:
	config = load_config()
	deck = PunchcardDeck(config['paths']['build'])

	if subset:

		loom_file = os.path.join(config['paths']['build'], "data", subset + ".loom")
		if os.path.exists(loom_file):
			merge_subset(subset, config)
			logging.info(f"Done.")
		else:
			logging.error(f"Loom file '{loom_file}' not found")

	else:
		exdir = os.path.abspath(os.path.join(config['paths']['build'], "merge"))
		cytograph_exe = shutil.which('cytograph')
		# Make directory for log files
		if not os.path.exists(exdir):
			os.mkdir(exdir)

		datadir = os.path.join(config['paths']['build'], "data")
		exportdir = os.path.join(config['paths']['build'], "exported")
		if not overwrite:
			logging.info("Rearranging directories...")
			shutil.copytree(datadir, os.path.join(config['paths']['build'], "data_premerge"))
			shutil.copytree(exportdir, os.path.join(config['paths']['build'], "exported_premerge"))

		logging.info("Submitting jobs")
		for subset in deck.get_leaves():
			# Use CPUs and memory from subset config
			config = load_config(subset)
			n_cpus = config.execution.n_cpus
			memory = config.execution.memory
			# Remove agg file and export directory
			task = subset.longname()
			os.remove(os.path.join(datadir, task + ".agg.loom"))
			shutil.rmtree(os.path.join(exportdir, task))
			# Make submit file
			cmd = f"merge --subset {task}"
			with open(os.path.join(exdir, task + ".condor"), "w") as f:
				f.write(f"""
	getenv       = true
	executable   = {os.path.abspath(cytograph_exe)}
	arguments    = "{cmd}"
	log          = {os.path.join(exdir, task)}.log
	output       = {os.path.join(exdir, task)}.out
	error        = {os.path.join(exdir, task)}.error
	request_cpus = {n_cpus}
	request_memory = {memory * 1024}
	queue 1\n
	""")
			# Submit
			subprocess.run(["condor_submit", os.path.join(exdir, task + ".condor")])

		# Check if merges are complete
		done = False
		while not done:
			time.sleep(30)
			logging.info("Checking merges...")
			done = True
			for subset in deck.get_leaves():
				f = os.path.join(exdir, "plots", f'{subset.longname()}.png')
				if not os.path.exists(f):
					done = False

		# Reaggregate and generate export folders
		logging.info("Processing new build...")
		subprocess.run(["cytograph", "build", "--engine", "condor"])
