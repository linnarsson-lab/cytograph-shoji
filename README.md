
# cytograph-shoji

## Installation

The following instructions should work for Linux and Mac (unfortunately, we have no 
experience with Windows).

1. [Install Anaconda](https://www.continuum.io/downloads)

2. Install [Shoji](https://github.com/linnarsson-lab/shoji)

3. Install OpenTSNE from source

```
git clone https://github.com/pavlin-policar/openTSNE.git
pip install -e openTSNE
```

3. Install `cytograph-shoji`:

```
git clone https://github.com/linnarsson-lab/cytograph-shoji.git
pip install -e cytograph-shoji
```

### Rebuilding the docs

Make sure you have [pdoc3](https://pdoc3.github.io/pdoc/) installed. In the repo, run:

```
pdoc --html cytograph --force
```

This will regenerate the API docs. Next, 



### Troubleshooting
If, when importing cytograph in python, you get errors related to imports from 'harmony', solve by:
```
pip install harmony-pytorch
```
(further reading on https://pypi.org/project/harmony-pytorch/)

```
If OpenTSNE does not use more than one CPU on macOS, try installing llvm using Homebrew:
```
brew install llvm
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export PATH="/usr/local/opt/llvm/bin:$PATH"
```
