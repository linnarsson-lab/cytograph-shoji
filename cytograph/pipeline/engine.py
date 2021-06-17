import logging
import os
import shutil
import subprocess
import sys
from typing import List, Dict, Set
from pathlib import Path

from .config import Config
from .punchcards import PunchcardDeck


class Engine:
	'''
	An execution engine, which takes a :class:`PunchcardDeck` and calculates an execution plan in the form
	of a dependency graph. The Engine itself does not actually execute the graph. This is the job of
	subclasses such as :class:`LocalEngine`` and :class:`CondorEngine`, which take the execution plan and
	executes it in some manner (e.g. locally and serially, or on a cluster and in parallel).
	'''

	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		self.deck = deck
		self.dryrun = dryrun
	
	def build_execution_dag(self) -> Dict[str, List[str]]:
		"""
		Build an execution plan in the form of a dependency graph, encoded as a dictionary.

		Returns:
			Dictionary mapping tasks to their dependencies
	
		Remarks:
			The tasks and their dependencies are named for the punchcards
		"""
		tasks: Dict[str, List[str]] = {}
		for name, punchcard in self.deck.punchcards.items():
			punchcard_sources = []
			for source in punchcard.sources:
				if source in self.deck.punchcards:
					punchcard_sources.append(source)
			tasks[name] = punchcard_sources
		return tasks
	
	def execute(self) -> None:
		pass


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
	result: List[str] = []
	seen: Set[str] = set()

	def recursive_helper(node: str) -> None:
		for neighbor in graph.get(node, []):
			if neighbor not in seen:
				seen.add(neighbor)
				recursive_helper(neighbor)
		if node not in result:
			result.append(node)

	for key in graph.keys():
		recursive_helper(key)
	return result


class LocalEngine(Engine):
	"""
	An execution engine that executes tasks serially and locally.
	"""
	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		super().__init__(deck, dryrun)

	def execute(self) -> None:
		# Run all the punchcards
		tasks = self.build_execution_dag()
		for task, deps in tasks.items():
			if len(deps) > 0:
				logging.debug(f"Task {task} depends on {','.join(deps)}")
			else:
				logging.debug(f"Task {task} has no dependencies")

		# Figure out a linear execution order consistent with the DAG
		ordered_tasks = topological_sort(tasks)
		logging.debug(f"Execution order: {','.join(ordered_tasks)}")

		# Now we have the tasks ordered by the DAG, and run them
		if self.dryrun:
			logging.info("Dry run only, with the following execution plan")
		for ix, task in enumerate(ordered_tasks):
			if not self.dryrun:
				logging.info(f"\033[1;32;40mBuild step {ix + 1} of {len(ordered_tasks)}: cytograph process {task}\033[0m")
				subprocess.run(["cytograph", "--hide-message", "process", task])
			else:
				logging.info(f"cytograph process {task}")


class CondorEngine(Engine):
	"""
	An engine that executes tasks in parallel on a HTCondor cluster, using the DAGman functionality
	of condor. Tasks will be executed in parallel as much as possible while respecting the
	dependency graph.
	"""
	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		super().__init__(deck, dryrun)

	def execute(self) -> None:
		config = Config().load()
		tasks = self.build_execution_dag()

		logdir: Path = config["paths"]["build"] / "logs"
		if logdir.exists():
			logging.warn("Removing previous build logs from 'logs' directory.")
			shutil.rmtree(logdir, ignore_errors=True)
		logdir.mkdir(exist_ok=True)

		# Find cytograph
		cytograph_exe = shutil.which('cytograph')
		if cytograph_exe is None:
			logging.error("The 'cytograph' command-line tool was not found.")
			sys.exit(1)

		for task in tasks.keys():
			if task not in self.deck.punchcards:
				logging.error(f"Punchcard {task} not found.")
				sys.exit(1)

		for task in tasks.keys():
			# Atomically check if the task has already been launched
			try:
				(logdir / (task + ".created")).touch(exist_ok=False)
			except FileExistsError:
				logging.info(f"Skipping '{task}' because it was already run (remove '{task}.created' from logs to force rebuild).")
				continue
			config = Config().load()  # Load it fresh for each task since we're clobbering it below
			cmd = ""

			punchcard = self.deck.punchcards[task]
			config = Config().load(punchcard)
			cmd = f"process {task}"
			# Must set 'request_gpus' only if non-zero, because even asking for zero GPUs requires a node that has GPUs (weirdly)
			request_gpus = f"request_gpus = {config['execution']['n_gpus']}" if config['execution']['n_gpus'] > 0 else ""
			with open(logdir / (task + ".condor"), "w") as f:
				f.write(f"""
getenv       = true
executable   = {os.path.abspath(cytograph_exe)}
arguments    = "{cmd}"
log          = {logdir / task}.log
output       = {logdir / task}.out
error        = {logdir / task}.error
request_cpus = {config["execution"]["n_cpus"]}
{request_gpus}
request_memory = {config["execution"]["memory"] * 1024}
queue 1\n
""")

		with open(os.path.join(logdir, "_dag.condor"), "w") as f:
			for task in tasks.keys():
				f.write(f"JOB {task} {logdir / task}.condor DIR {config['paths']['build']}\n")
			for task, deps in tasks.items():
				if len(deps) == 0:
					continue
				f.write(f"PARENT {' '.join(deps)} CHILD {task}\n")

		if not self.dryrun:
			logging.debug(f"condor_submit_dag {logdir / '_dag.condor'}")
			subprocess.run(["condor_submit_dag", logdir / "_dag.condor"])
		else:
			logging.info(f"(Dry run) condor_submit_dag {logdir / '_dag.condor'}")

# TODO: SlurmEngine using job dependencies (https://hpc.nih.gov/docs/job_dependencies.html)
# TODO: SgeEngine using job dependencies (https://arc.leeds.ac.uk/using-the-systems/why-have-a-scheduler/advanced-sge-job-dependencies/)
