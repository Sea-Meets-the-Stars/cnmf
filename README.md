# cnmf

## Constrained Non-negative Matrix Factorization -- mainly for Inherent Optical Properties (IOPs)

### Brought to you by J. Xavier Prochaska (UC Santa Cruz)

## Installing

1. Clone the repository: git clone https://github.com/AI-for-Ocean-Science/cnmf.git
1. Change into the directory: cd cnmf
1. Install the package: pip install -e ".[dev]"

This will install most of the packages required for the package to run. 

## Additional Dependencies (not installed by pip via the setup.py)

* [oceancolor](https://github.com/AI-for-Ocean-Science/ocean-color.git)

## See this paper for an application of CNMF to absorption coefficients, with Patrick Gray (University of Maine):

[On the Fundamental Additive Modes of Ocean Color Absorption](https://doi.org/10.22541/essoar.171828481.15444713/v1)

### To run code related to that paper, you will need to download data from Loisel et al. (2023).  Follow these instructions:

1. Grab the Loisel+2023 files from [this link on Dryad](https://datadryad.org/stash/dataset/doi:10.6076/D1630T)
2. Put the file `Hydrolight400.nc` in a folder on your system named `some_path/data/loisel2023/`
3. Point the environment variable `OS_COLOR` to the path `some_path/`
4. e.g. in your .bashrc or .bash_profile, add the line:
```export OS_COLOR=/home/xavier/Projects/Oceanography/Color/```