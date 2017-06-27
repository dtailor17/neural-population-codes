# neural-population-codes
Code accompanying dissertation for MSc in Artificial Intelligence

**Project title:** The effect of noise correlations and heterogeneity on the emergence of biases in multivariate neural population codes.

**Supervisor:** Dr. Mark van Rossum, University of Edinburgh
___

## Environment setup

Miniconda with Python 2.7

Packages installed:
* numpy
* scipy
* matplotlib
* jupyter
* seaborn
* dill

Package list also exported to *conda_packages.txt*; to reproduce environment use `conda create`.

## IPython notebook

Notebook session state serialised (to preserve results). Uncompress *session_state.pkl.gz* first using `gunzip` followed by executing `dill.load_session('session_state.pkl')` within the notebook.
