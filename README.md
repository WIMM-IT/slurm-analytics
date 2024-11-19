# Slurm Analytics

Analyse Slurm sacct data with Python Pandas.

#### Analysis categories

- resource use. Are users wasting the CPUs or memory they request?
  - CPU consumption and % utilisation
  - Memory consumption and % utilisation
  - user breakdown of "resource waste"

- wait times. Does cluster have enough resources for demand?
  - weekly distribution of job wait times [as % of job time limit]
  - total requested resources waiting as % of partition capacity
  - wait time vs recent resource use

- users. Who are the biggest users?
  - weekly plots of most active users according to different metrics: CPUs, memory, #jobs, elapsed time
  - summary plot of biggest users (CPU & memory)

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

`sacct` output is parsed using the parser taken from [AWS's HPC Cost Simulator](https://github.com/aws-samples/hpc-cost-simulator):

    git clone https://github.com/aowenson-imm/hpc-cost-simulator

`git clone` my fork, add the parent folder to `PYTHONPATH`, then run:

    python sacct-parse.py -n <cluster-name>

This saves the parsed data, as Python pickled Pandas DataFrames, in local folder `Parsed/<cluster-name>/`

Python requirements:
- pandas
- numpy
- GitPython

## Step 3: analyse

    usage: analyse.py [-h] -n CLUSTER_NAME [-r] [-w] [-u] [-t WEEKS]
                      [-p PARTITION]

    Analyse sacct data, generating plots.

    optional arguments:
      -h, --help            show this help message and exit
      -r, --resources       Analyse resource use
      -w, --waits           Analyse wait times
      -u, --users           Analyse users
      -p PARTITION, --partition PARTITION
                            Analyse just this Slurm partition
      -a, --annual          Compare years (only some charts supported)
      -n WEEKS, --weeks WEEKS
                            Analyse just last N weeks
      -t {H,D,W}, --resolution {H,D,W}
                            Aggregation resolution time period, default: D
      -d PLOTS_DIR, --plots-dir PLOTS_DIR
                            Save plots here instead of "Plots/"

Python requirements:
- pandas
- numpy
- matplotlib
- numba

Aggregation is the core function that combines one particular resource attribute e.g. MaxRSS from all jobs to a single time-series of any time resolution e.g. 1 second or 1 day. Performance optimised with Numba, and aggregation results cached in local folder `Cache`.

The plot images are written to local folder `Plots/`.
Maybe in future the plots can be generated/viewed via a React webpage etc.
