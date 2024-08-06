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

## Step 3: analyse

This is the fun part.

    usage: analyse.py [-h] -n CLUSTER_NAME [-r] [-w] [-u] [-t WEEKS]
                      [-p PARTITION]

    Analyse sacct data, generating plots.

    optional arguments:
      -h, --help            show this help message and exit
      -n CLUSTER_NAME, --cluster-name CLUSTER_NAME
                            Cluster name
      -r, --resources       Analyse resource use
      -w, --waits           Analyse wait times
      -u, --users           Analyse users
      -t WEEKS, --weeks WEEKS
                            Analyse just last N weeks
      -p PARTITION, --partition PARTITION
                            Analyse just this Slurm partition

Python requirements:
- pandas
- numpy
- matplotlib
- hashlib  # caching
- numba  # optimise core loop

The key part is function `aggregate_resource`. 
This aggregates one particular resource attribute e.g. `MaxRSS` from all jobs to a single time-series of any time resolution e.g. 1 second or 1 day.
`numba` optimises the critical inner aggregation loop.
Performance improved further by caching aggregation results in local folder `Cache` - cache key is the `hashlib` sha256 of the input DataFrame object.

There are 3 analysis categories:

- resource use. Are users wasting the CPUs or memory they request?

- wait times. Do most users wait for little time? Is wait time getting worse over time? Simply too many users for a particular partition?

- users. Who are the power users?

The plot images are written to local folder `Plots/`.
