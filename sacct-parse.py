#!/usr/bin/python3

import pandas as pd
import numpy as np
import datetime as _dt
import os
import glob
import pickle as pkl
import json
import re
from pprint import pprint

# https://github.com/aowenson-imm/hpc-cost-simulator/tree/slurm-sacct-parser
import SlurmLogParser


import argparse
parser = argparse.ArgumentParser(description="Parse sacct output")
parser.add_argument('-n', '--cluster-name', type=str, required=True, help='Cluster name')
args = parser.parse_args()
cluster_id = args.cluster_name


df_all = None
input_fps = []
pattern = f'Dumps/{cluster_id}/sacct/*.csv'
for input_fp in glob.glob(pattern):
  input_fps.append(input_fp)
input_fps = sorted(input_fps, reverse=True)  # parse newest first, in case user aborts early
if len(input_fps) == 0:
  print(f"No sacct files detected for cluster named '{cluster_id}'")
  quit()


pattern = os.path.join('Dumps', cluster_id, 'resources', 'partitions-????-??-??.json')
with open(sorted(glob.glob(pattern))[-1]) as f:
  resources = json.load(f)
partitions = list(resources.keys())


pkl_all_fp = os.path.join('Parsed', cluster_id, 'sacct-output-combined.pkl')


if os.path.isfile(pkl_all_fp):
  df_all_rebuild = False
  with open(pkl_all_fp, 'rb') as F:
    df_all = pkl.load(F)
    if df_all is None or df_all.empty:
      df_all_rebuild = True
else:
  df_all_rebuild = True


if len(input_fps) > 0:
  for i in range(len(input_fps)):
    fp = input_fps[i]
    with open(fp, "r") as f:
      nlines = sum(1 for _ in f)
    if nlines == 1:
      # Just header
      continue

    fn = os.path.basename(fp)
    pkl_fp = os.path.join('Parsed', cluster_id, fn).replace(".csv", ".pkl")
    df = None
    re_parse = False
    if os.path.isfile(pkl_fp):
      pkl_mod_dt = os.path.getmtime(pkl_fp)
      csv_mod_dt = os.path.getmtime(fp)
      re_parse = csv_mod_dt > pkl_mod_dt
      if not (re_parse or df_all_rebuild):
        continue
      if df_all_rebuild and not re_parse:
        # Need the data
        with open(pkl_fp, 'rb') as F:
          df = pkl.load(F)

    df_was_None = df is None
    if df is None:
      print(f"Parsing: {fp}")
      pattern = r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})'
      match = re.search(pattern, fp)
      if match:
          start, end = match.groups()
      else:
          start, end = None, None
      slurmLogParser = SlurmLogParser.SlurmLogParser(fp, starttime=start, endtime=end)
      d = slurmLogParser.parse_jobs_to_dict()
      if d is None or len(d) == 0:
        # continue
        raise Exception(f'Parsing file created 0 jobs: {fp}')
      df = pd.DataFrame.from_dict(d, orient='index')
      # raise Exception(df.columns)
      for c in list(df.columns):
        if c.endswith('_dt') or c.endswith('_td'):
          c2 = c[:-3]
          if c2 in df.columns:
            df = df.drop(c2, axis=1).rename(columns={c:c2})

      # Process phantom partitions:
      # - dump-cluster-resources.py prunes "meta" partitions
      # - also handle old partitions that were renamed/shuffled - their nodes are mapped to current partitions
      f = ~df['Partition'].isin(partitions)
      if f.any():
        bads_parts = df.loc[f,'Partition'].unique()
        part_remaps = {}
        for badp in bads_parts:
          badp_nodes = df['NodeList'][df['Partition']==badp].unique()
          for n in badp_nodes:
            key = (badp, n)
            f = (df['Partition']==badp) & (df['NodeList']==n)
            if key not in part_remaps:
              candidiates = [p for p in partitions if n in resources[p]['Nodes']]
              if len(candidiates) == 0:
                if n == 'None assigned':
                  # These jobs never ran, so hopefully the analysis will already be discarding these.
                  part_remaps[key] = '_phantom_'
                else:
                  raise Exception(f"Need to remap: part '{badp}' on node '{n}'")
              elif len(candidiates) == 1:
                part_remaps[key] = candidiates[0]
              else:
                # Pick candidate with most resources.
                newp = candidiates[0]
                for c in candidiates[1:]:
                  if resources[newp]['MaxCPUsPerNode']*len(resources[newp]['Nodes']) < \
                    resources[c]['MaxCPUsPerNode']*len(resources[c]['Nodes']):
                    newp = c
                part_remaps[key] = newp
            df.loc[f, 'Partition'] = part_remaps[key]
      f = ~df['Partition'].isin(partitions+['_phantom_'])
      if f.any():
        raise Exception(f'Bad partitions still present: {df["Partition"][f].unique()}')

       # Clean
      df = df.dropna(axis=1, how='all')
      for c in df.columns:
        if (df[c]=='').all():
          df = df.drop(c, axis=1)

      f_bad = df['MaxRSS'].isna().to_numpy() & (df['State']=='COMPLETED').to_numpy()
      if f_bad.any():
        # In our Slurm setup, when a job has finished, then sacct will output 2 rows:
        #     <JOBID> ...
        #     <JOBID>.batch ...
        # Each row has some of the job data, so we need both rows.
        # But in rare cases, sacct does not output the second line despite job completing weeks before.
        # A subsequent call can succeed.
        # Clue that second row is missing is parsed 'MaxRSS' is none/empty.
        # So drop these rows.
        print(f"- dropping {np.sum(f_bad)} incomplete jobs")
        df = df[~f_bad]

      f_running = df['State'] == 'RUNNING'
      if f_running.any():
        df = df[~f_running]

      df = df.sort_values('Submit')

      for c in ['MaxRSS']:
        df[c+' GB'] = df[c] * 1e-9
        df = df.drop(c, axis=1)

    # Add to combined table
    if df_all is None:
      if df is None:
        raise Exception('why df None?')
      df_all = df
    else:
      f_overlap = df.index.isin(df_all.index)
      if f_overlap.any():
        # Seems sacct filters on start <= time <= end, 
        # not start <= time < end. So a job can appear
        # in adjacent time ranges if end time exactly equals boundary.
        f_time0 = df['End'].dt.time == _dt.time(0)
        f_overlap_bad = f_overlap & (~f_time0)
        df_new = df[~f_overlap]
      else:
        df_new = df
      df_all = pd.concat([df_all, df_new]) ; df_all = df_all.sort_values('Submit')

    # Persist df_all
    if not os.path.isdir(os.path.dirname(pkl_all_fp)):
      os.makedirs(os.path.dirname(pkl_all_fp))
    with open(pkl_all_fp, 'wb') as F:
      pkl.dump(df_all, F)

    if df_was_None or re_parse:
      # Persist 'df'. Important to do *after* updating 'df_all',
      # as existence of 'df' prevents re-parsing this file.
      with open(pkl_fp, 'wb') as F:
        pkl.dump(df, F)
