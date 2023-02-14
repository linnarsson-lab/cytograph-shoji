import os
import re
import subprocess
import numpy as np
import inspect
import importlib
from .algorithm import Algorithm


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
		c[~np.isfinite(c)] = 0  # -inf inf NaN
	return c
	
def export_loompy(ws, path,expression_tensor='Expression'):
    """Export a workspace to a loom file"""
    import loompy
    ra_attrs = {'Gene':ws.Gene[:]}
    ca_attrs = {'Clusters':ws.Clusters[:],'MolecularNgh':ws.MolecularNgh[:],'X':ws.X[:],'Y':ws.Y[:],'Sample':ws.Sample[:], 'Embedding':ws.Embedding[:]}
    expression = ws[expression_tensor][:]
    loompy.create(path, expression.T, row_attrs=ra_attrs, col_attrs=ca_attrs)
    
# https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
def available_cpu_count():
	""" Number of available virtual or physical CPUs on this system, i.e.
	user/real as output by time(1) when called with an optimally scaling
	userspace-only program"""

	# cpuset
	# cpuset may restrict the number of *available* processors
	try:
		m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', open('/proc/self/status').read())
		if m:
			res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
			if res > 0:
				return res
	except IOError:
		pass

	# Python 2.6+
	try:
		import multiprocessing
		return multiprocessing.cpu_count()
	except (ImportError, NotImplementedError):
		pass

	# https://github.com/giampaolo/psutil
	try:
		import psutil
		return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
	except (ImportError, AttributeError):
		pass

	# POSIX
	try:
		res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

		if res > 0:
			return res
	except (AttributeError, ValueError):
		pass

	# Windows
	try:
		res = int(os.environ['NUMBER_OF_PROCESSORS'])

		if res > 0:
			return res
	except (KeyError, ValueError):
		pass

	# BSD
	try:
		sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'], stdout=subprocess.PIPE)
		scStdout = sysctl.communicate()[0]
		res = int(scStdout)

		if res > 0:
			return res
	except (OSError, ValueError):
		pass

	# Linux
	try:
		res = open('/proc/cpuinfo').read().count('processor\t:')

		if res > 0:
			return res
	except IOError:
		pass

	# Solaris
	try:
		pseudoDevices = os.listdir('/devices/pseudo/')
		res = 0
		for pd in pseudoDevices:
			if re.match(r'^cpuid@[0-9]+$', pd):
				res += 1

		if res > 0:
			return res
	except OSError:
		pass

	# Other UNIXes (heuristic)
	try:
		try:
			dmesg = open('/var/run/dmesg.boot').read()
		except IOError:
			dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
			dmesg = dmesgProcess.communicate()[0]

		res = 0
		while '\ncpu' + str(res) + ':' in dmesg:
			res += 1

		if res > 0:
			return res
	except OSError:
		pass

	raise Exception('Can not determine number of CPUs on this system')

def shoji2anndata(ws,save=False):
    import scanpy as sc
    shape = ws.Expression.shape
    X = ws.Expression[:]
    gene_shape = ws.Gene.shape
    cell_shape = (X.shape[0],)
    
    attrs = [x for x in dir(ws) if x[0] != '_' and x[0].isupper()]
    obs = {}
    #obsm = {}
    var = {}
    for att in attrs:
        if ws[att].shape == gene_shape:
            print('Added attribute {} to AnnData vars'.format(att))
            var[att] = ws[att][:]
        elif ws[att].shape == cell_shape:
            print('Added attribute {} to AnnData obs'.format(att))
            obs[att] = ws[att][:]
        else:
            print('Attribute {} NOT added to AnnData'.format(att),ws[att].shape)
            
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.var.index = adata.var.Gene
    adata.var_names_make_unique()
    if save:
        adata.write_h5ad(save+'.h5ad')
    return adata
    
        
    
def get_decorators(function):
	"""Returns list of decorators names

	Args:
		function (Callable): decorated method/function

	Return:
		List of decorators as strings

	Example:
		Given:

		@my_decorator
		@another_decorator
		def decorated_function():
			pass

		>>> get_decorators(decorated_function)
		['@my_decorator', '@another_decorator']

	"""
	source = inspect.getsource(function)
	index = source.find("def ")
	return [
		line.strip()
		for line in source[:index].strip().splitlines()
		if line.strip()[0] == "@"
	]


def deindent(s):
	lines = s.split("\n")
	n_tabs = 0
	for c in lines[1]:
		if c == "\t":
			n_tabs += 1
		else:
			break
	return "\n".join(line[n_tabs:] for line in lines)


def make_cytograph_docs():
	modules = {}
	for cls in sorted(Algorithm.__subclasses__(), key=lambda cls: ".".join(cls.__module__.split(".")[:2])):
		name = ".".join(cls.__module__.split(".")[:2])
		modules.setdefault(name, [])
		modules[name].append(cls)

	aside = """
		<aside class="bd-aside sticky-xl-top text-muted align-self-start mb-3 mb-xl-5 px-2">
			<h2 class="h6 pt-4 pb-3 mb-4 border-bottom">Algorithms</h2>
			<nav class="small" id="toc">
				<ul class="list-unstyled">
	"""
	for mod, classes in modules.items():
		aside += f"""
			<li class="my-2">
				<button class="btn d-inline-flex align-items-center collapsed" data-bs-toggle="collapse" aria-expanded="false" data-bs-target="#contents-collapse" aria-controls="contents-collapse">{mod}</button>
				<ul class="list-unstyled ps-3 collapse" id="contents-collapse">
		"""
		for cls in classes:
			alg = cls.__name__
			aside += f"""
			<li><a class="d-inline-flex align-items-center rounded" href="#{mod}.{alg}">{alg}</a></li>
			"""
		aside += "</ul></li>"
	aside += "</ul></nav></aside>"

	main = """
		<div class="bd-cheatsheet container-fluid bg-body">
		<section id="content">
	"""

	for mod, classes in modules.items():
		main += f"<h2>Module '{mod}'</h2>"
		module = importlib.import_module('cytograph.' + cls.__module__.split(".")[1])
		if module.__doc__ is not None:
			main += "<p>" + deindent(module.__doc__) + "</p>"

		for cls in classes:
			alg = cls.__name__
			main += f"<h3 style='color: rgb(25, 118, 210)'><a id='{mod}.{alg}'>{alg}</a></h3>"
			if cls.__doc__ is not None:
				main += "<p>" + deindent(cls.__doc__) + "</p>"

			decs = get_decorators(cls.fit)
			if any("@requires" in d for d in decs):
				main += "<strong>Required tensors</strong>"
				main += "<table class='table table-striped'><tr><th></th><th>dtype</th><th>dims</th></tr>"
				for dec in decs:
					if "#" in dec:
						dec = dec[dec.rindex("#") - 1:]
					if "@requires" in dec:
						items = dec[dec.index("("):dec.rindex(")")].split(",")
						main += "<tr><td>" + items[0].strip('"()\' ') + "</td><td>" + items[1].strip('"()\' ') + "</td><td>" + dec[dec.index(",", dec.index(",") + 1):-1].strip('",\' ') + "</td><tr/>"
				main += "</table>"

			if any("@creates" in d for d in decs):
				main += "<strong>Created tensors</strong>"
				main += "<table class='table table-striped'><tr><th></th><th>dtype</th><th>dims</th></tr>"
				for dec in decs:
					if "#" in dec:
						dec = dec[dec.rindex("#") - 1:]
					if ", indices=True" in dec:
						dec = dec[:dec.index(", indices=True")]
					if "@creates" in dec:
						items = dec[dec.index("("):dec.rindex(")") + 1].split(",")
						main += "<tr><td>" + items[0].strip('"()\' ') + "</td><td>" + items[1].strip('"()\' ') + "</td><td>" + dec[dec.index(",", dec.index(",") + 1):-1].strip('",\' ') + "</td><tr/>"
				main += "</table>"

			main += f"Recipe: <strong>{alg}</strong>:" + " {"
			main += ", ".join([f"<strong>{parm.name}</strong>: {str(parm).split(':')[1] if ':' in str(parm) else 'Any'}" for name, parm in inspect.signature(cls.__init__).parameters.items() if name not in ("self", "args", "kwargs")])
			main += "}<br/>"
			main += "<p>&nbsp;</p>"

			if cls.__init__.__doc__ is not None:
				main += "<pre>" + deindent(cls.__init__.__doc__) + "</pre>"
			main += "<p>&nbsp;</p>"
			main += "<p>&nbsp;</p>"

	main += "</section></div>"
	return """
		<!doctype html>
		<html lang="en">
			<head>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<meta name="author" content="Sten Linnarsson">
				<title>Cytograph algorithms</title>
				<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
				<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
				<style>
					body {
						display: grid;
						gap: 1rem;
						grid-template-columns: 4fr 1fr;
						grid-template-rows: auto;
					}
					.bd-aside {
						padding-top: 4rem;
						grid-area: 1 / 2;
						scroll-margin-top: 4rem;
					}
				</style>
			</head>
		</html>
		<body class='bg-light'>
	""" + aside + main + """
		</body></html>
	"""
