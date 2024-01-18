# ThesisSpace

This repository reflects the work for the Master Thesis of Thomas Bosboom with Stefano Speretta as supervisor.

The code has the following structure:
- The folder [report](report/) contains the contents related to the report of thesis work.
- The folder [simulations](simulations/) contains the underlying code that is used in the report. Within this folder, one can find the actual source code and the code used for testing and plotting in [src](simulations/src/) and [tests](simulations/tests/) respectively. These folders contain information used from reference data. This can be found in [reference](simulations/reference/).

For proper installation follow this guide. The installation of tudatpy is supported exclusively through the use of a conda package manager,
such as Miniconda or Anaconda. For new users, and Windows users in particular, the use of Anaconda is recommended.

Download the environment setup file "environment.yaml". Then, in your terminal navigate to the directory containing this file and execute the following command:

```
conda env create -f environment.yaml
```
With the conda environment now installed, you can activate it to work in it using:

```
conda activate tudat-space
```

For more information, go to one of the documentation of tudatpy.

- [Main Documentation](https://docs.tudat.space/en/latest/)
- [API Reference](https://py.api.tudat.space/en/latest/index.html)
