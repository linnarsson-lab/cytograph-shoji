
# cytograph

## Installation

The following instructions should work for Linux and Mac (unfortunately, we have no 
experience with Windows).

1. [Install Anaconda](https://www.continuum.io/downloads), Python 3.7 version

2. Install Shoji

3. Install OpenTSNE from source (git clone, pip install -e .)

```
git clone https://github.com/pavlin-policar/openTSNE.git
cd openTSNE
pip install -e .
```

3. Install `cytograph`:

```
git clone https://github.com/linnarsson-lab/cytograph-dev.git
cd cytograph-dev
pip install -e .
```

### Troubleshooting
If, when importing cytograph in python, you get errors related to imports from 'harmony', solve by:
```
pip install harmony-pytorch
```
(further reading on https://pypi.org/project/harmony-pytorch/)

Errors related to 'numba' package, e.g. during HPF/PCA generation. Try (possibly downgrading):
```
conda update anaconda
conda install numba=0.46.0
```

If OpenTSNE does not use more than one CPU on macOS, try installing llvm using Homebrew:

```
brew install llvm
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export PATH="/usr/local/opt/llvm/bin:$PATH"
```
