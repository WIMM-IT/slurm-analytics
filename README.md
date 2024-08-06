# Slurm Analytics

Analyse Slurm sacct data with Python Pandas.

# Step 1: extract `sacct` data

`sacct-dump.py` is a lightweight Python script to output data from sacct:
- no 3rd-party dependencies
- outputs 1 week on each call to minimise Slurm load

Saves in local folder `Dumps/<cluster-name>/sacct/`.
Analyses contents to decide what week to fetch next.

This script will also run `dump-cluster-resources.py`, 
that saves node resource information into `Dumps/<cluster-name>/resources/`.
