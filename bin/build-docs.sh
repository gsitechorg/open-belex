# Usage: from the belex directory, type ./bin/build-docs.sh
# ASSUMES that pyenv is installed on the local machine.
# ASSUMES that Sphinx has been set up with separate build
# and source directories

# Adding docs: Incrementally add docstrings to classes,
# functions, methods. Use numpydoc-style docstrings. Use
# module "apl_optimizations.py" for examples: it's the
# one with the most manually-added docstrings. Most of
# the other modules are autodocced.

# see   ../belex/docs/sphinx/source:
# conf.py and index.rst for foundational information

pushd docs/sphinx  # from /belex

pyenv local 3.9.7
python -m venv sphinx-venv
source ./sphinx-venv/bin/activate
pip install -U sphinx numpy hypothesis myst-parser cerberus mermaid sphinxcontrib-mermaid sphinx-rtd-theme numpydoc

pushd  # back to /belex
pip install -e .
pushd  # back to /belex/docs/sphinx

mkdir -p source/_static
mkdir -p source/_templates
make clean
make html
make latex

pushd build/latex
make clean
make
popd  # back to /belex/docs/sphinx

popd  # back to /belex
