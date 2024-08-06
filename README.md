# Slurm Analytics

Analyse Slurm sacct data with Python Pandas.

## Step 1: extract `sacct` data

Run `python sacct-dump.py` in the cluster, as any user.
This is a lightweight Python script to output data from sacct:
- no 3rd-party dependencies
- outputs 1 week on each call to minimise Slurm load

Saves in local folder `Dumps/<cluster-name>/sacct/`.
Analyses contents to decide what week to fetch next.

This script will also run `dump-cluster-resources.py`, 
that saves node resource information into `Dumps/<cluster-name>/resources/`.

## Step 2: parse `sacct` data

`sacct` output is parsed using the parser taken from AWS's HPC Cost Simulator:

    https://github.com/aowenson-imm/hpc-cost-simulator/tree/slurm-sacct-parser

`git clone` my fork, add the parent folder to `PYTHONPATH`, then run:

    python sacct-parse.py -n <cluster-name>

This saves the parsed data, as Python pickled Pandas DataFrames, in local folder `Parsed/<cluster-name>/`
