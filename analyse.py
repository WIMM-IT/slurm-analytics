#!/usr/bin/python3


import pandas as pd
import numpy as np
import os
import glob
import pickle as pkl
import hashlib
from pprint import pprint

# Optimise
from numba import jit
# from time import perf_counter as pc

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib
plt.rcParams['figure.autolayout'] = True  # auto tight layout
# marker = 'x'
marker = '.'
fig_dims = (14, 8)


import argparse
parser = argparse.ArgumentParser(description="Analyse sacct data, generating plots.")
parser.add_argument('cluster_name', type=str, help='Cluster name')
parser.add_argument('-r', '--resources',  action='store_true', help='Analyse resource use')
parser.add_argument('-w', '--waits',      action='store_true', help='Analyse wait times')  # Working on a metric for "queue fairness"
parser.add_argument('-u', '--users',      action='store_true', help='Analyse users')
parser.add_argument('-p', '--partition',  type=str, default=None, help='Analyse just this Slurm partition')
parser.add_argument('-n', '--weeks',      type=int, default=None, help='Analyse just last N weeks')
parser.add_argument('-t', '--resolution', type=str, default='D', choices=['H', 'D', 'W'], help='Aggregation resolution time period, default: %(default)s')
parser.add_argument('-d', '--plots-dir', type=str, default='Plots', help='Save plots here instead of "Plots/"')
args = parser.parse_args()
cluster_id = args.cluster_name
plot_dp = os.path.join(args.plots_dir, cluster_id)

if not args.resources and not args.waits and not args.users:
  print("You did not specify an analysis")
  quit()


import json
pattern = os.path.join('Dumps', cluster_id, 'resources', 'partitions-????-??-??.json')
with open(sorted(glob.glob(pattern))[-1]) as f:
  resources = json.load(f)
partitions = list(resources.keys())
total_gb = {p: resources[p]['MaxMemPerNode']*len(resources[p]['Nodes'])/1000 for p in resources}
total_gb['*'] = sum(total_gb.values())
total_cpus = {p: resources[p]['MaxCPUsPerNode']*len(resources[p]['Nodes']) for p in resources}
total_cpus['*'] = sum(total_cpus.values())

_total_cpus, _total_memory, _total_nodes = 0, 0, 0
for part in resources.values():
    n = len(part['Nodes'])
    _total_cpus += part['MaxCPUsPerNode'] * n
    _total_memory += part['MaxMemPerNode'] * n
    _total_nodes += n
avg_cpus = _total_cpus / _total_nodes
avg_memory = _total_memory / _total_nodes


# Input file. If not exist, then run 'slurm-usage-dump.py' on Slurm server.
pkl_all_fp = os.path.join("Parsed", cluster_id, "sacct-output-combined.pkl")
if not os.path.isfile(pkl_all_fp):
  raise Exception('Run sacct-parse.py first')

with open(pkl_all_fp, 'rb') as F:
  df = pkl.load(F)


# Clean cache of old data
if os.path.isdir('Cache'):
  input_mod_dt = os.path.getmtime(pkl_all_fp)
  for fn in os.listdir('Cache'):
    fp = os.path.join('Cache', fn)
    if os.path.isfile(fp):
        file_mod_dt = os.path.getmtime(fp)
        if file_mod_dt < input_mod_dt:
            os.remove(fp)


if args.weeks:
  cutoff_dt = df['End'].max() - pd.Timedelta(weeks=args.weeks)
  df = df[df['End']>=cutoff_dt]


# Clean data:
df = df[df['Elapsed'] > pd.Timedelta(0)]
df = df.drop('CPUTimeRAW', axis=1, errors='ignore')  # pointless because CPUTimeRaw * NCPUS = Elapsed


# This is the critical magic function, for combining all jobs in df
# to a single time-series.
def aggregate_resource(df, x, period='H', cache=True):
  # Map job-start-time to a short intervals. Similarly for job-end.
  if not isinstance(df, pd.DataFrame):
    raise ValueError(f'"df" must be a Pandas DataFrame not {type(df)}')
  if not isinstance(x, str):
    raise ValueError(f'"x" must be a string not {type(x)}')
  if not isinstance(period, str):
    raise ValueError(f'"period" must be a string not {type(period)}')

  if cache:
    cache_dir = "Cache"
    x_hash = hashlib.sha256(df[x].values.tobytes()).hexdigest()[:16]
    cache_key = f"{x.replace(' ', '_')}_{period}_{x_hash}"
    cache_path = os.path.join(cache_dir, cache_key+".pkl")
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.isfile(cache_path):
      # See if can reuse cache
      cache_mod_dt = os.path.getmtime(cache_path)
      input_mod_dt = os.path.getmtime(pkl_all_fp)
      if cache_mod_dt > input_mod_dt:
        # Reuse cache
        with open(cache_path, 'rb') as f:
            return pkl.load(f)

  fna = df[x].isna()
  if fna.any():
    df = df[~fna]

  # print(f"Aggregating '{x}' on '{period}' period")
  if period == 'H':
    freq='H'
    time_index = pd.date_range(start=df['Start'].min().floor('H'), end=df['End'].max().ceil('H'), freq=freq)
  elif period == 'D':
    freq='D'
    time_index = pd.date_range(start=df['Start'].min().floor('D'), end=df['End'].max().ceil('D'), freq=freq)
  elif period == '15m':
    freq = '15T'
    time_index = pd.date_range(start=df['Start'].min().floor('H'), end=df['End'].max().ceil('H'), freq=freq)
  elif period == '5m':
    freq = '5T'
    time_index = pd.date_range(start=df['Start'].min().floor('H'), end=df['End'].max().ceil('H'), freq=freq)
  elif period == '1m':
    freq = '1T'
    time_index = pd.date_range(start=df['Start'].min().floor('H'), end=df['End'].max().ceil('H'), freq=freq)
  else:
    raise Exception(f"Not implemented period='{period}'")
  itd = time_index[1] - time_index[0]
  intervals_df = pd.DataFrame({
    'IntStart': time_index[:-1],
    'IntEnd': time_index[1:]})
  intervals_df.index = intervals_df['IntStart']
  intervals_df.index.names = ['idx']
  df = df.sort_values('Start')
  df2 = pd.merge_asof(df, intervals_df, left_on='Start', right_on='IntStart', direction='backward')
  df2.index = df.index
  df = df2
  df = df.rename({'IntStart':'Start_IntStart', 'IntEnd':'Start_IntEnd'}, axis=1)
  df = df.sort_values('End')
  df2 = pd.merge_asof(df, intervals_df, left_on='End', right_on='IntStart', direction='backward')
  df2.index = df.index
  df = df2
  df = df.rename({'IntStart':'End_IntStart', 'IntEnd':'End_IntEnd'}, axis=1)


  # Bin resource use use into intervals
  df['Start_Idx'] = intervals_df.index.get_indexer(df['Start_IntStart'])
  df['End_Idx'] = intervals_df.index.get_indexer(df['End_IntStart'])
  df['NI'] = (df['End_Idx'] - df['Start_Idx']) + 1

  # 1) handle start & end intervals for each job.
  # 1a) calculate % of start & end intervals that job consumed
  f = (df['Start_Idx'] == df['End_Idx']).to_numpy()
  if f.any():
    # Some jobs ran within a single interval so can't assume these will touch interval boundaries.
    df['StartOverlap%'] = 0.0 ; df['EndOverlap%'] = 0.0
    df.loc[f,'StartOverlap%'] = (df.loc[f,'End'] - df.loc[f,'Start']) / itd

    # But for all other jobs, these span at least 2 intervals so they cross interval boundaries.
    fn = ~f
    df.loc[fn, 'StartOverlap%'] = (df.loc[fn, 'Start_IntEnd'] - df.loc[fn, 'Start']) / itd
    df.loc[fn, 'EndOverlap%'] = (df.loc[fn, 'End'] - df.loc[fn, 'End_IntStart']) / itd
  else:
    df['StartOverlap%'] = (df['Start_IntEnd'] - df['Start']) / itd
    df['EndOverlap%'] = (df['End'] - df['End_IntStart']) / itd
  # 1b) used above % to distribute resource use across interval
  if x == 'N':
    # special:
    df[f'Start_Int_{x}'] = 1.0 * df['StartOverlap%']
    df[f'End_Int_{x}'] = 1.0 * df['EndOverlap%']
  else:
    df[f'Start_Int_{x}'] = df[x] * df['StartOverlap%']
    df[f'End_Int_{x}'] = df[x] * df['EndOverlap%']
  agg_start = df.groupby('Start_IntStart').agg(
      xsum=pd.NamedAgg(column=f'Start_Int_{x}', aggfunc='sum')
  )
  agg_end = df.groupby('End_IntStart').agg(
      xsum=pd.NamedAgg(column=f'End_Int_{x}', aggfunc='sum')
  )
  agg_start.index.names = ['IntStart']
  agg_start = agg_start['xsum']
  agg_end.index.names = ['IntStart']
  agg_end = agg_end['xsum']
  agg = pd.Series(index=time_index, data=0)
  agg = agg.add(agg_start, fill_value=0)
  agg = agg.add(agg_end, fill_value=0)

  # 2) handle the "inner intervals", for those jobs spanning more than 2 intervals.
  f = df['End_Idx'] > (df['Start_Idx']+1)
  if f.any():
    inner_df = df[f].copy()  # massive perf improvement for coarse aggregation (most jobs in 1 interval)
    # inner_df = df.copy()
    inner_df['Inner_StartIdx'] = inner_df['Start_Idx'] + 1
    inner_df['Inner_EndIdx'] = inner_df['End_Idx'] - 1

    # agg_backup = agg.copy()
    # # Have to manually iterate over each job, but use indices to optimise updating sum.
    # inner_df = inner_df.sort_values('Elapsed')
    # # for i in range(inner_df.shape[0]):
    # t = tqdm.tqdm(range(inner_df.shape[0]))
    # t.set_description(f'Aggregating {x} ...')
    # t0 = pc()
    # for i in t:
    #   dfi = inner_df.iloc[i]
    #   # if x != 'N' and np.isnan(dfi[x]):
    #   #   print(dfi)
    #   #   raise Exception(f'NaNs detected at i={i}')
    #   # Update: exception has not triggered for long time.
    #   start = dfi['Inner_StartIdx']
    #   end = dfi['Inner_EndIdx']
    #   if start > end:
    #     # ToDo: this could be most jobs, so potential performance optimisation here
    #     continue
    #   startidx = agg.index[start]
    #   endidx = agg.index[end]
    #   if x == 'N':
    #     agg.loc[startidx:endidx] += 1.0
    #   else:
    #     agg.loc[startidx:endidx] += dfi[x]
    # t1 = pc()
    # print(f"- old time = {t1-t0:.2} seconds")

    # Optimised with Numba:
    # t0 = pc()
    @jit(nopython=True)
    def process_inner_intervals(start_idx_values, end_idx_values, x_values, agg_values):
      for i in range(len(start_idx_values)):
        start_idx = start_idx_values[i]
        end_idx = end_idx_values[i]
        if start_idx <= end_idx:
          agg_values[start_idx:end_idx+1] += x_values[i]
      return agg_values
    Inner_StartIdx_values = inner_df['Inner_StartIdx'].to_numpy()
    Inner_EndIdx_values = inner_df['Inner_EndIdx'].to_numpy()
    if x == 'N':
        x_values = np.ones(len(inner_df))
    else:
        x_values = inner_df[x].to_numpy()
    agg_values = agg.to_numpy()
    # agg_values = agg_backup.to_numpy()
    # print(f'Aggregating {x} (period={period})')
    agg_values2 = process_inner_intervals(Inner_StartIdx_values, 
                                          Inner_EndIdx_values, 
                                          x_values, 
                                          agg_values)
    # agg2 = pd.Series(index=time_index, data=agg_values2)
    agg = pd.Series(index=time_index, data=agg_values2)
    # t1 = pc()
    # print(f"- JIT time = {t1-t0:.2} seconds")
    # print("- agg2.equals(agg) =", agg2.equals(agg))
    # if not agg2.equals(agg):
    #   raise Exception("JIT output is different")
    # else:
    #   print("GOOD, JIT matches")

  agg.name = f'{x}_sum'

  # Even with JIT, caching still helps but speedup much less (~1.5x)
  if cache:
    with open(cache_path, 'wb') as f:
      pkl.dump(agg, f)

  return agg


# For scatter plots with a skewed-to-left distribution, use
# this function instead.
def plot_skewed_binned_sum(df, x, y, nbins=50):
  fig, ax = plt.subplots(figsize=(14, 8))

  # Plot sqrt of data, to prevent big values distorting chart.
  # Better than log scaling.
  df_plt = df[[x, y]].copy()
  x2 = x + ' sqrt'
  df_plt[x2] = np.sqrt(df_plt[x])

  # Bin data & plot
  bin_edges = np.linspace(df_plt[x2].min(), df_plt[x2].max(), nbins + 1)
  df_plt['bin'] = pd.cut(df_plt[x2], bins=bin_edges, include_lowest=True, labels=False)
  binned_sum = df_plt.groupby('bin')[y].sum()
  binned_sum = binned_sum.reindex(range(nbins), fill_value=0)
  bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
  plt.bar(bin_centers, binned_sum.to_numpy(), width=np.diff(bin_edges), align='center', edgecolor='black')
  plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda z, _: int(z**2)))
  plt.xlabel(x)
  plt.ylabel(y)

  return fig, ax


def plot_time_series_violins(df, x, s, period='W', yscaling=None):
  if df.empty:
    raise Exception('empty')

  df_plt = df[[x, s]].copy()
  df_plt = df_plt[df_plt[x] > 0.0]
  if df_plt.empty:
    print(df[[x, s]])
    raise Exception(f'No non-zero data to plot from column "{x}"')

  if yscaling:
    if yscaling == 'log':
      x2 = x + ' log'
      df_plt[x2] = np.log(df_plt[x])
    else:
      x2 = x + ' sqrt'
      df_plt[x2] = np.sqrt(df_plt[x])
  else:
    x2 = x

  df_plt = df_plt.resample(period).apply(list)
  df_plt = df_plt[df_plt[x].apply(lambda z: len(z) > 1)]

  # Create plot
  fig, ax = plt.subplots(figsize=(24, 8))
  
  # Use column 's' to scale width of violins, so
  # area is proportional to e.g. total system load during period.
  df_plt[s] = [np.sum(l) for l in df_plt[s]]
  df_plt[s] = df_plt[s] / df_plt[s].max()  # normalise 's'

  iqrs = []
  global_min = None
  for i in range(df_plt.shape[0]):
    l = df_plt[x2].iloc[i]
    l = sorted(l)
    q1, q3 = np.percentile(l, [15, 100-15])
    iqr = q3 - q1
    if iqr == 0.0:
      # Default to minimum X value
      if global_min is None:
        global_min = min([min(df_plt[x2].iloc[j]) for j in range(len(df_plt))])
      iqr = global_min
    iqrs.append(iqr)
  widths = np.divide(df_plt[s], iqrs)
  widths = widths / np.max(widths)  # normalise
  df_plt['Widths'] = widths

  # - calculate max width
  positions = []
  for i, (chunk, data) in enumerate(df_plt.iterrows()):
    pos = mdates.date2num(chunk) + 2.0  # convert date to epoch for X-axis
    positions.append(pos)
  diffs = np.diff(sorted(positions))
  max_width = np.min(diffs) if len(diffs) > 0 else 1
  # - plot
  for i, (chunk, data) in enumerate(df_plt.iterrows()):
    load = data[s]
    if load < 0.00001:
      continue
    pos = positions[i]
    times = data[x2]
    values = np.hstack(times)
    #
    width = data['Widths']
    proportional_width = width * max_width
    vp = ax.violinplot(values, positions=[pos], widths=proportional_width)
    for body in vp['bodies']:
      body.set_alpha(1)
    vp['cbars'].set_alpha(0.0)
    for k in ['cmins', 'cmaxes']:
      vp[k].set_alpha(0.3)

  # Convert X-axis epochs to date strings
  ax.xaxis.set_major_locator(mdates.WeekdayLocator())
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.xticks(rotation=45)

  if yscaling:
    yticks = plt.yticks()[0]
    if yscaling == 'log':
      original_values = np.exp(yticks)
    else:
      original_values = np.square(yticks)
      ax.set_ylim([0, None])
    plt.yticks(yticks, np.round(original_values, 1))
    if yscaling == 'sqrt':
      ax.set_ylim([0, None])

  return fig, ax


def fn_analyse_resources(df):
  for p in [args.partition] if args.partition else partitions+['*']:
    if p == '*':
      df_plt = df
    else:
      df_plt = df[df['Partition']==p].copy()
      if df_plt.empty:
        continue
    if p == '*':
      print(f"Analysing resource use across all partitions")
    else:
      print(f"Analysing resource use in partition '{p}'")

    os.makedirs(plot_dp, exist_ok=True)
    if p != '*':
      os.makedirs(os.path.join(plot_dp, p.upper()), exist_ok=True)

    # Memory first
    mem_col = 'MaxRSS GB'
    rss = aggregate_resource(df_plt, mem_col, args.resolution)
    req = aggregate_resource(df_plt, 'ReqMemGB', args.resolution)
    df_mem = pd.DataFrame(rss).join(req)
    for x in list(df_mem.columns):
      df_mem[f'{x} %'] = df_mem[f'{x}'] / total_gb[p]
    # - absolute time-series plot
    fig = plt.figure(figsize=(14, 8))
    y_used = df_mem[f'{mem_col}_sum'].copy()
    y_req = df_mem['ReqMemGB_sum'].copy()
    unit = 'GB'
    if y_req.max() > 1200:
      y_used *= 0.001
      y_req *= 0.001
      unit = 'TB'
      if y_req.max() > 1200:
        y_used *= 0.001
        y_req *= 0.001
        unit = 'EB'
    plt.fill_between(df_mem.index, y_used, label='Used', color='orange')
    plt.fill_between(df_mem.index, y_req, y_used, label='Requested', color='blue')
    plt.xlabel('Date')
    plt.ylabel(f'Memory {unit}')
    plt.title(f'Memory utilisation - partition {p}')
    plt.legend()
    fn = 'resource-memory.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)
    # - % time-series plot
    fig = plt.figure(figsize=(14, 8))
    y_used = df_mem[f'{mem_col}_sum %']*100.0
    y_req = df_mem['ReqMemGB_sum %']*100.0
    plt.fill_between(df_mem.index, y_used, label='Used', color='orange')
    plt.fill_between(df_mem.index, y_req, y_used, label='Requested', color='blue')
    plt.xlabel('Date')
    plt.ylabel(f'Memory % partition')
    plt.ylim(0, 100)
    plt.title(f'Memory utilisation % - partition {p}')
    plt.legend()
    fn = 'resource-memory-pct.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)

    # Now analyse CPU
    df_plt['AvgCPU'] = df_plt['TotalCPU'].to_numpy() / df_plt['NCPUS'].to_numpy()
    df_plt['AvgCPULoad'] = df_plt['AvgCPU'] / df_plt['Elapsed']
    df_plt['NCPUS_real'] = df_plt['NCPUS'].astype('float') * df_plt['AvgCPULoad']
    req = aggregate_resource(df_plt, 'NCPUS', args.resolution)
    real = aggregate_resource(df_plt, 'NCPUS_real', args.resolution)
    df_cpu = pd.DataFrame(real).join(req)
    for x in list(df_cpu.columns):
      df_cpu[f'{x} %'] = df_cpu[f'{x}'] / total_cpus[p]
    # - absolute time-series plot
    fig = plt.figure(figsize=(14, 8))
    y_req = df_cpu['NCPUS_sum']
    y_used = df_cpu['NCPUS_real_sum']
    plt.fill_between(df_cpu.index, y_used, label='Used', color='orange')
    plt.fill_between(df_cpu.index, y_req, y_used, label='Requested', color='blue')
    plt.xlabel('Date')
    plt.ylabel(f'CPUs')
    plt.title(f'CPU utilisation - partition {p}')
    plt.legend()
    fn = 'resource-cpus.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)
    # - % time-series plot
    fig = plt.figure(figsize=(14, 8))
    y_req = df_cpu['NCPUS_sum %']*100.0
    y_used = df_cpu['NCPUS_real_sum %']*100.0
    plt.fill_between(df_cpu.index, y_used, label='Used', color='orange')
    plt.fill_between(df_cpu.index, y_req, y_used, label='Requested', color='blue')
    plt.xlabel('Date')
    plt.ylabel(f'CPUs % partition')
    plt.ylim(0, 100)
    plt.title(f'CPU utilisation - partition {p}')
    plt.legend()
    fn = 'resource-cpus-pct.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)

    # Now calculate % wasted for both CPU and memory, and combine into single chart
    df_mem['Mem wasted'] = df_mem['ReqMemGB_sum'] - df_mem[f'{mem_col}_sum']
    df_cpu['CPU wasted'] = df_cpu['NCPUS_sum'] - df_cpu['NCPUS_real_sum']
    # - first chart = waste as % of total system resource
    df_sys_waste = df_mem.join(df_cpu)
    df_sys_waste['Mem wasted %'] = df_sys_waste['Mem wasted'] / total_gb[p]
    df_sys_waste['CPU wasted %'] = df_sys_waste['CPU wasted'] / total_cpus[p]
    fig = plt.figure(figsize=(14, 8))
    if df_sys_waste['Mem wasted %'].mean() > df_sys_waste['CPU wasted %'].mean():
      # Fill CPU plot, draw memory on top as line
      plt.plot(df_sys_waste.index, df_sys_waste['Mem wasted %']*100.0, label='Memory', color='blue')
      plt.fill_between(df_sys_waste.index, df_sys_waste['CPU wasted %']*100.0, label='CPU', color='orange')
    else:
      # Fill memory plot, draw CPU on top as line
      plt.plot(df_sys_waste.index, df_sys_waste['CPU wasted %']*100.0, label='CPU', color='blue')
      plt.fill_between(df_sys_waste.index, df_sys_waste['Mem wasted %']*100.0, label='Memory', color='orange')
    plt.xlabel('Time')
    plt.ylabel(f'System waste %')
    # plt.ylim(0, 100)
    plt.title(f'System waste - partition {p}')
    plt.legend()
    fn = 'resource-waste.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)

    # Analyse most-wasteful users
    df_plt['Elapsed seconds'] = df_plt['Elapsed'].dt.total_seconds()
    df_plt['Elapsed hours'] = df_plt['Elapsed seconds'] * (1.0 / (60*60))
    # - memory
    df_mem_users = df_plt[['User', 'Elapsed hours', mem_col, 'ReqMemGB']].copy()
    df_mem_users[f'{mem_col}-hours'] = df_mem_users[mem_col] * df_mem_users['Elapsed hours']
    df_mem_users['ReqMemGB-hours'] = df_mem_users['ReqMemGB'] * df_mem_users['Elapsed hours']
    df_mem_users = df_mem_users.drop(['Elapsed hours', mem_col, 'ReqMemGB'], axis=1)
    df_mem_users['Wasted GB-hours'] = df_mem_users['ReqMemGB-hours'] - df_mem_users[f'{mem_col}-hours']
    df_mem_users['Wasted TB-hours'] = df_mem_users['Wasted GB-hours'] * 0.001
    df_mem_users_top = df_mem_users.groupby('User').sum().sort_values('Wasted GB-hours', ascending=False)
    df_mem_users_top = df_mem_users_top[df_mem_users_top['Wasted TB-hours'] > 100]
    top_wasteful_users = df_mem_users_top[df_mem_users_top['Wasted TB-hours'] > 100].index
    if len(top_wasteful_users) > 0:
      if len(top_wasteful_users) > 8:
        # Constrained by number of distinct colours for chart
        top_wasteful_users = top_wasteful_users[:8]
      #
      fig = plt.figure(figsize=(14, 8))
      stackplot_data = None
      for u in top_wasteful_users:
        dfu = df_plt[df_plt['User']==u].copy()

        rss = aggregate_resource(dfu, mem_col, args.resolution)
        req = aggregate_resource(dfu, 'ReqMemGB', args.resolution)
        dfu_mem = pd.DataFrame(rss).join(req)
        dfu_mem['Mem wasted'] = dfu_mem['ReqMemGB_sum'] - dfu_mem[f'{mem_col}_sum']
        dfu_mem['Mem wasted'] = dfu_mem['Mem wasted'].clip(lower=0.0)
        dfu_mem['Mem wasted %'] = dfu_mem['Mem wasted'] / total_gb[p] *100.0

        dfu_mem = dfu_mem[['Mem wasted %']].rename(columns={'Mem wasted %':u}).copy()
        if stackplot_data is None:
          stackplot_data = dfu_mem
        else:
          stackplot_data = stackplot_data.join(dfu_mem).fillna(0)
      plt.stackplot(stackplot_data.index.date, [stackplot_data[c] for c in stackplot_data.columns], labels=stackplot_data.columns)
      plt.xlabel('Time')
      plt.ylabel(f'Memory waste %')
      plt.title(f'Memory waste - partition {p}')
      plt.legend()
      fn = 'resource-waste-memory-user-breakdown.png'
      plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
      plt.savefig(plt_fp)
      plt.close(fig)
    # - CPU
    df_cpu_users = df_plt[['User', 'Elapsed hours', 'NCPUS', 'NCPUS_real']].copy()
    df_cpu_users['CPU-hours'] = df_cpu_users['NCPUS'] * df_cpu_users['Elapsed hours']
    df_cpu_users['CPU_real-hours'] = df_cpu_users['NCPUS_real'] * df_cpu_users['Elapsed hours']
    df_cpu_users = df_cpu_users.drop(['Elapsed hours', 'NCPUS', 'NCPUS_real'], axis=1)
    df_cpu_users['Wasted CPU-hours'] = df_cpu_users['CPU-hours'] - df_cpu_users['CPU_real-hours']
    df_cpu_users_top = df_cpu_users.groupby('User').sum().sort_values('Wasted CPU-hours', ascending=False)
    df_cpu_users_top = df_cpu_users_top[df_cpu_users_top['Wasted CPU-hours'] > 10000]
    top_wasteful_users = df_cpu_users_top[df_cpu_users_top['Wasted CPU-hours'] > 10000].index
    if len(top_wasteful_users) > 0:
      if len(top_wasteful_users) > 8:
        # Constrained by number of distinct colours for chart
        top_wasteful_users = top_wasteful_users[:8]
      #
      fig = plt.figure(figsize=(14, 8))
      stackplot_data = None
      for u in top_wasteful_users:
        dfu = df_plt[df_plt['User']==u].copy()

        ncpus_req = aggregate_resource(dfu, 'NCPUS', args.resolution)
        ncpus_real = aggregate_resource(dfu, 'NCPUS_real', args.resolution)
        dfu_cpu = pd.DataFrame(ncpus_req).join(ncpus_real)
        dfu_cpu['CPUs wasted'] = dfu_cpu['NCPUS_sum'] - dfu_cpu['NCPUS_real_sum']
        dfu_cpu['CPUs wasted'] = dfu_cpu['CPUs wasted'].clip(lower=0.0)
        dfu_cpu['CPUs wasted %'] = dfu_cpu['CPUs wasted'] / total_cpus[p] *100

        dfu_cpu = dfu_cpu[['CPUs wasted %']].rename(columns={'CPUs wasted %':u}).copy()
        if stackplot_data is None:
          stackplot_data = dfu_cpu
        else:
          stackplot_data = stackplot_data.join(dfu_cpu).fillna(0)
      plt.stackplot(stackplot_data.index.date, [stackplot_data[c] for c in stackplot_data.columns], labels=stackplot_data.columns)
      plt.xlabel('Time')
      plt.ylabel(f'CPUs waste %')
      plt.legend()
      plt.title(f'CPUs waste - partition {p}')
      fn = 'resource-waste-cpus-user-breakdown.png'
      plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
      plt.savefig(plt_fp)
      plt.close(fig)


def fn_analyse_waiting(df):
  # Clean data:
  # df = df.drop('State', axis=1, errors='ignore')
  df = df.drop('AvgCPU', axis=1, errors='ignore')
  df = df.drop('MinCPU', axis=1, errors='ignore')
  df = df.drop('CPUTimeRAW', axis=1, errors='ignore')
  df = df.drop('MaxRSS GB', axis=1, errors='ignore')

  # Exclude jobs with artificial user-created delays
  artificial_delays = {'Dependency', 'DependencyNeverSatisfied', 'BeginTime', 'JobHeldUser', 'ReqNodeNotAvail'}
  f_user_delay = (df['Reason'].isin(artificial_delays)).to_numpy()
  if f_user_delay.any():
    # print(f"Dropping {np.sum(f_user_delay)}/{df.shape[0]} jobs as these had artificial delay caused by user")
    df = df[~f_user_delay].copy()

  df['Timelimit seconds'] = df['Timelimit'].dt.total_seconds()
  df['Timelimit days'] = df['Timelimit seconds'] * (1.0 / (24*60*60))
  df['Timelimit hours'] = df['Timelimit seconds'] * (1.0 / (60*60))

  df['WaitTime'] = df['Start'] - df['Submit']
  df['WaitTime seconds'] = df['WaitTime'].dt.total_seconds()
  df['WaitTime hours'] = df['WaitTime seconds'] * (1.0 / (60*60))
  df['WaitTime minutes'] = df['WaitTime hours']*60
  df['WaitTime days'] = df['WaitTime seconds'] * (1.0 / (24*60*60))

  df['Elapsed seconds'] = df['Elapsed'].dt.total_seconds()
  df['Elapsed hours'] = df['Elapsed seconds'] * (1.0 / (60*60))
  df['Elapsed minutes'] = df['Elapsed hours']*60

  df['WaitTime % elapsed'] = df['WaitTime hours'] / df['Elapsed hours']
  df['WaitTime % limit'] = df['WaitTime hours'] / df['Timelimit hours']

  df['TotalCPU seconds'] = df['TotalCPU'].dt.total_seconds()
  df['TotalCPU hours'] = df['TotalCPU seconds'] * (1.0 / (60*60))
  df['ReqMem TB hours'] = df['ReqMemGB'] * df['Elapsed hours'] * 0.001

  # Round some numbers
  df['WaitTime hours'] = df['WaitTime hours'].round(1)
  df['TotalCPU hours'] = df['TotalCPU hours'].round(1)
  df['ReqMem TB hours'] = df['ReqMem TB hours'].round(1)


  df_wait = df[['Submit', 'Start', 'Partition', 'WaitTime hours', 'WaitTime % limit', 'NCPUS', 'ReqMemGB', 'TotalCPU hours', 'Elapsed hours']].copy()

  df_wait['TB*CPU'] = df_wait['ReqMemGB'] * 0.001 * df_wait['NCPUS']
  rcol = 'TB*CPU'

  # Time-series for each partition
  for p in [args.partition] if args.partition else partitions:
    if p == '*':
      print(f"Analysing wait time across all partitions")
      dfp = df_wait.copy()
    else:
      dfp = df_wait[df_wait['Partition']==p].copy()
    dfp = dfp[dfp['WaitTime hours']>0]
    dfp = dfp[dfp['WaitTime hours']<48]
    if dfp.empty:
      continue
      print(f"Analysing wait time in partition '{p}'")

    os.makedirs(plot_dp, exist_ok=True)
    if p != '*':
      os.makedirs(os.path.join(plot_dp, p.upper()), exist_ok=True)

    load_metric = 'TotalCPU hours'
    dfps = dfp.set_index('Submit')

    # Violin plot of binned wait times
    df_plot = dfp[['Submit', 'WaitTime hours', 'WaitTime % limit', load_metric]].copy().set_index('Submit')
    df_plot['WaitTime % limit'] = 100*df_plot['WaitTime % limit']
    # Absolute wait time
    if df_plot['WaitTime hours'].max() > 40:
      fig, ag = plot_time_series_violins(df_plot, 'WaitTime hours', load_metric, yscaling='log')
    elif df_plot['WaitTime hours'].max() > 40:
      fig, ag = plot_time_series_violins(df_plot, 'WaitTime hours', load_metric, yscaling='sqrt')
    else:  
      fig, ag = plot_time_series_violins(df_plot, 'WaitTime hours', load_metric)
    plt.xlabel('Week end')
    plt.ylabel('Wait time (hours)')
    plt.title(f'Wait time, weekly distributions, area ≈ {load_metric}, partition {p}')
    fn = 'wait-distribution-weekly.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)
    #
    # Relative wait time (% of time-limit)
    if df_plot['WaitTime % limit'].max() > 3000:
      # Need more aggressive Y scaling
      fig, ag = plot_time_series_violins(df_plot, 'WaitTime % limit', load_metric, yscaling='log')
    else:
      fig, ag = plot_time_series_violins(df_plot, 'WaitTime % limit', load_metric, yscaling='sqrt')
    plt.xlabel('Week end')
    plt.ylabel('Wait time % time limit')
    #
    plt.title(f'Wait time % job limit, weekly distributions, area ≈ {load_metric}, partition {p}')
    fn = 'wait-distribution-weekly-pct.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)


    # Plot of submit time vs wait
    # Update: this plot redundant, the violin plot above much better
    # x = dfp[['Submit', 'WaitTime hours']].copy()
    # # f='4H'
    # f='1D'
    # x = x.groupby(pd.Grouper(key='Submit', freq=f)).describe()['WaitTime hours']
    # f_na = x['mean'].isna()
    # if f_na.any():
    #   x = x[~f_na]
    # # - Interquartile plot
    # x = x[['mean', '25%', '75%', 'max']]
    # mean_values = x['mean']
    # q1_values = x['25%']
    # q3_values = x['75%']
    # # - calculate the interquartile range
    # lower_error = np.clip(mean_values - q1_values, 0, None)
    # upper_error = np.clip(q3_values - mean_values, 0, None)
    # asymmetric_error = [lower_error, upper_error]
    # # - create the plot
    # fig = plt.figure(figsize=(14, 6))
    # plt.errorbar(x.index, mean_values, yerr=asymmetric_error, fmt='none', label='IQR', capsize=5, color='blue')
    # plt.scatter(x.index, mean_values, marker='_', color='red', label='Mean', zorder=5)
    # # plt.scatter(x.index, x['max'], marker='_', color='blue', label='Max')
    # plt.xlabel('Submit time')
    # plt.ylabel('Wait duration (hours)')
    # plt.title(f'Submit time vs wait time - partition {p}')
    # plt.legend()
    # # plt.show() ; quit()
    # # plt_fp = os.path.join(plot_dp, f'submit-vs-wait-{p.upper()}.png')
    # fn = 'wait-vs-submit.png'
    # plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    # plt.savefig(plt_fp)
    # plt.close(fig)


    # Plot of resources waiting in queue
    df_wait2 = dfp.rename(columns={'Start':'End'}).rename(columns={'Submit':'Start'})
    # - CPUS
    ncpus_waiting = aggregate_resource(df_wait2, 'NCPUS', period=args.resolution)
    ncpus_waiting = pd.DataFrame(ncpus_waiting)
    for x in list(ncpus_waiting.columns):
      ncpus_waiting[f'{x} %'] = ncpus_waiting[f'{x}'] / total_cpus['*']
    # - time-series plot
    fig = plt.figure(figsize=(14, 8))
    # Plot normal % chart
    y = ncpus_waiting['NCPUS_sum %']*100.0
    # plt.plot(ncpus_waiting.index, y, label='Requested')
    plt.plot(ncpus_waiting.index, y, label='Requested', color='blue')
    plt.fill_between(ncpus_waiting.index, y, alpha=0.3, color='blue')
    plt.ylim(0, min(400, y.max()))  # cap otherwise chart is not readable
    plt.ylabel(f'CPUs % partition')
    plt.title(f'CPUs waiting % partition "{p}" resources')
    plt.xlabel('Date')
    plt.legend()
    fn = 'waiting-cpus-pct.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)
    # - memory
    mem_waiting = aggregate_resource(df_wait2, 'ReqMemGB', period=args.resolution)
    mem_waiting = pd.DataFrame(mem_waiting)
    for x in list(mem_waiting.columns):
      mem_waiting[f'{x} %'] = mem_waiting[f'{x}'] / total_gb['*']
    # - time-series plot
    fig = plt.figure(figsize=(14, 8))
    y = mem_waiting['ReqMemGB_sum %']*100.0
    plt.plot(mem_waiting.index, y, label='Requested', color='orange')
    plt.fill_between(mem_waiting.index, y, alpha=0.3, color='orange')
    plt.ylim(0, min(400, y.max()))  # cap otherwise chart is not readable
    plt.ylabel(f'Memory waiting % partition')
    plt.title(f'Memory waiting % partition "{p}" resources')
    plt.xlabel('Date')
    plt.legend()
    fn = 'waiting-mem-pct.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)
    # - combine
    cpus_mem_waiting = ncpus_waiting.join(mem_waiting)
    fig = plt.figure(figsize=(14, 8))
    y_cpu = cpus_mem_waiting['NCPUS_sum %']*100.0
    y_mem = cpus_mem_waiting['ReqMemGB_sum %']*100.0
    plt.plot(cpus_mem_waiting.index, y_mem, label='Memory', color='orange')
    plt.plot(cpus_mem_waiting.index, y_cpu, label='CPU', color='blue')
    plt.ylabel(f'Resources waiting % partition')
    plt.ylim(0, min(400, max(y_cpu.max(), y_mem.max())))  # cap otherwise chart is not readable
    plt.xlabel('Date')
    plt.title(f'Resources waiting % partition "{p}" resources')
    plt.legend()
    fn = 'waiting-cpus-mem-pct.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)

  return

  p = '*'

  ################################################################
  # EXPERIMENTAL: TRY TO FIND A PLOT TO REPRESENT FAIRNESS
  ################################################################
  # period = 'H'   ; window = 6
  # df_wait['Time floored 1h'] = df_wait['Submit'].dt.floor('1H')
  # period = '15m' ; window = 4*6
  # df_wait['Time floored 15m'] = df_wait['Submit'].dt.floor('15T')
  # period = '5m'  ; window = 12*6
  # df_wait['Time floored 5m'] = df_wait['Submit'].dt.floor('5T')
  period = '1m'  ; window = 60
  df_wait['Time floored 1m'] = df_wait['Submit'].dt.floor('1T')

  fc = [c for c in df_wait.columns if 'Time floored' in c][0]
  df_wait['Recent use'] = np.nan
  for u in df_wait['User'].unique():
    df_wait_user = df_wait[df_wait['User']==u].drop('User', axis=1)
    rcol_agg = aggregate_resource(df_wait_user, rcol, period=period)
    df_wait_user_agg = pd.DataFrame(rcol_agg)
    df_wait_user_agg['Recent use'] = df_wait_user_agg[rcol+'_sum'].rolling(window=window, min_periods=1).sum()
    df_wait_user_agg = df_wait_user_agg.drop(rcol+'_sum', axis=1)
    # Merge:
    df_wait_user_agg.index.name = fc
    df_wait_user_agg = df_wait_user_agg.reset_index()
    df_wait_user_agg['User'] = u
    merged = df_wait.merge(df_wait_user_agg, on=['User', fc], how='left')
    c = 'Recent use'
    f_na = df_wait[c].isna().to_numpy()
    df_wait.loc[f_na, c] = merged[c+'_y'].to_numpy()[f_na]
  df_wait = df_wait.sort_values('WaitTime hours')
  fig = plt.figure(figsize=(16,10))
  plt.scatter(df_wait['Recent use'], df_wait['WaitTime hours'], s=1,
              c=df_wait['User'].astype('category').cat.codes, cmap='viridis')
  plt.xlabel(f'Recent use: {rcol}')
  plt.ylabel('Future wait hours')
  fn = 'wait-vs-recent-use.png'
  plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
  plt.savefig(plt_fp)
  plt.close(fig)
  ################################################################

  return

  ################################################################
  ## THESE PLOTS PROBABLY NOT USEFUL
  ################################################################
  # Time-series across all partitions
  df_plt = df[['Submit', 'Partition', 'WaitTime hours', 'ReqMemGB', 'TotalCPU hours', 'Elapsed hours']].copy()
  df_plt = df_plt.set_index('Submit')
  df_plt['Load'] = df_plt['ReqMemGB'] * df_plt['TotalCPU hours'] * df_plt['Elapsed hours']
  df_plt = df_plt.drop(['ReqMemGB', 'TotalCPU hours', 'Elapsed hours'], axis=1)
  fig, ag = plot_time_series_violins(df_plt, 'WaitTime hours', 'Load')
  plt.xlabel('Week end')
  plt.ylabel('Wait time (hours)')
  plt.title('Wait time distributions trend (area ≈ load)')
  fp = os.path.join(plot_dp, 'wait-distribution-trend.png')
  plt.savefig(fp)
  plt.close(fig)
  # One violin plot of all partitions
  fig = plt.figure(figsize=(14, 6))
  x = df_plt['WaitTime hours']
  x = x[x>0]
  xt = np.sqrt(x)
  plt.violinplot(xt)
  yticks = plt.yticks()[0]
  original_values = np.square(yticks)
  plt.yticks(yticks, np.round(original_values, 3))
  plt.xlabel('Frequency')
  plt.ylabel('Wait time (hours)')
  plt.title('Wait time distribution')
  fn = 'wait-distribution.png'
  plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
  plt.savefig(plt_fp)
  plt.close(fig)


  # Plot of request time vs wait
  fig, ax = plot_skewed_binned_sum(df, 'Timelimit days', 'WaitTime days')
  ax.set_xlabel('Time limit (days)')
  ax.set_ylabel('Sum wait-time (days)')
  ax.set_title('Time limit vs wait')
  fn = 'wait-vs-timelimitƒ.png'
  plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
  fig.savefig(plt_fp)
  plt.close(fig)

  # Plot of memory vs wait
  fig, ax = plot_skewed_binned_sum(df, 'ReqMemGB', 'WaitTime days')
  ax.set_xlabel('Requested GB')
  ax.set_ylabel('Sum wait-time (days)')
  ax.set_title('GB vs wait')
  fn = 'wait-vs-memory.png'
  plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
  fig.savefig(plt_fp)
  plt.close(fig)

  # Plot of CPUs vs wait
  fig, ax = plot_skewed_binned_sum(df, 'NCPUS', 'WaitTime days')
  ax.set_xlabel('Requested CPUs')
  ax.set_ylabel('Sum wait-time (days)')
  ax.set_title('CPUs vs wait')
  fn = 'wait-vs-cpus.png'
  plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
  fig.savefig(plt_fp)
  plt.close(fig)

  # Plot of wait-time, over time, for small jobs (<=100G) vs large
  df_plt = df[['Submit', 'WaitTime hours', 'ReqMemGB']]
  df_plt = df_plt[df_plt['WaitTime hours'] > 0]
  f_neg = df_plt['WaitTime hours'] <= 0.0
  if f_neg.any():
    print(df_plt[f_neg])
    raise Exception('investigate')
  fig = plt.figure(figsize=(14, 6))
  f_small = df_plt['ReqMemGB'] <= 100
  df_plt_small = df_plt[f_small]
  df_plt_big = df_plt[~f_small]
  f='W'
  # small
  x_small = df_plt_small.groupby(pd.Grouper(key='Submit', freq=f)).describe()['WaitTime hours']
  f_na = x_small['mean'].isna()
  if f_na.any():
    x_small = x_small[~f_na]
  # - Interquartile plot
  x_small = x_small[['mean', '25%', '75%', 'max']]
  mean_values = x_small['mean']
  q1_values = x_small['25%']
  q3_values = x_small['75%']
  # - calculate the interquartile range
  lower_error = np.clip(mean_values - q1_values, 0, None)
  upper_error = np.clip(q3_values - mean_values, 0, None)
  asymmetric_error = [lower_error, upper_error]
  # - create the plot
  x_small.index = x_small.index - pd.Timedelta(hours=12)
  plt.errorbar(x_small.index, mean_values, yerr=asymmetric_error, fmt='none', label='Small jobs', capsize=5, color='blue')
  plt.scatter(x_small.index, mean_values, marker='_', color='blue', label='_nolegend_', zorder=5)
  plt.scatter(x_small.index, x_small['max'], marker='_', color='blue', label='_nolegend_')
  # large
  x_big = df_plt_big.groupby(pd.Grouper(key='Submit', freq=f)).describe()['WaitTime hours']
  f_na = x_big['mean'].isna()
  if f_na.any():
    x_big = x_big[~f_na]
  # - Interquartile plot
  x_big = x_big[['mean', '25%', '75%', 'max']]
  mean_values = x_big['mean']
  q1_values = x_big['25%']
  q3_values = x_big['75%']
  # - calculate the interquartile range
  lower_error = np.clip(mean_values - q1_values, 0, None)
  upper_error = np.clip(q3_values - mean_values, 0, None)
  asymmetric_error = [lower_error, upper_error]
  # - create the plot
  x_big.index = x_big.index + pd.Timedelta(hours=12)
  plt.errorbar(x_big.index, mean_values, yerr=asymmetric_error, fmt='none', label='Large jobs', capsize=5, color='orange')
  plt.scatter(x_big.index, mean_values, marker='_', color='orange', label='_nolegend_', zorder=5)
  plt.scatter(x_big.index, x_big['max'], marker='_', color='orange', label='_nolegend_')
  # format
  plt.yscale('log')
  plt.xlabel('Submit datetime')
  plt.ylabel('Wait duration (hours)')
  plt.title(f'Submit vs wait - partition {p}')
  plt.legend()
  fn = 'wait-time-small-vs-large-jobs.png'
  plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
  plt.savefig(plt_fp)
  plt.close(fig)


def fn_analyse_users(df):
  df['Elapsed hours'] = df['Elapsed'].dt.total_seconds()/3600
  df['CPU hours'] = df['TotalCPU'].dt.total_seconds()/3600
  df['GB hours'] = df['Elapsed hours'] * df['MaxRSS GB']
  df['100GB hours'] = df['GB hours']*0.01
  df['TB hours'] = df['GB hours']*0.001

  df_plt = df[['User', 'Partition', 'Submit', 'GB hours', 'CPU hours', 'Elapsed hours']]

  for p in [args.partition] if args.partition else partitions+['*']:
    if p == '*':
      df_plt_p = df_plt
    else:
      df_plt_p = df_plt[df_plt['Partition']==p]
      if df_plt_p.empty:
        continue
    df_plt_p = df_plt_p.drop('Partition', axis=1)
    if p == '*':
      print(f"Analysing users across all partitions")
    else:
      print(f"Analysing users in partition '{p}'")

    os.makedirs(plot_dp, exist_ok=True)
    if p != '*':
      os.makedirs(os.path.join(plot_dp, p.upper()), exist_ok=True)

    df_plt_p = df_plt_p.set_index('Submit')
    users_sum = df_plt_p.groupby('User').sum()

    users_sum = users_sum.sort_values('GB hours', ascending=True)
    # discard users using very little, otherwise plot too crowded
    if len(users_sum) > 20:
      threshold = 0.01
      f_light = (users_sum['CPU hours'] < (users_sum['CPU hours'].max()*threshold))
      f_light = f_light & (users_sum['GB hours'] < (users_sum['GB hours'].max()*threshold))
      f_light = f_light & (users_sum['Elapsed hours'] < (users_sum['Elapsed hours'].max()*threshold))
      if f_light.any():
        users_sum = users_sum[~f_light]
    fig, ax = plt.subplots(figsize=(10, 12))
    indices = range(users_sum.shape[0])  # y-locations for the groups
    bar_height = 0.35
    if p == '*':
      ax.barh([i - bar_height / 2 for i in indices], users_sum['CPU hours'], bar_height, label='CPU')
      ax.barh([i + bar_height / 2 for i in indices], users_sum['GB hours'], bar_height, label='GB')
    else:
      ax.barh([i - bar_height / 2 for i in indices], users_sum['CPU hours']*(1/resources[p]['MaxCPUsPerNode']), bar_height, label='Node-CPUs')
      ax.barh([i + bar_height / 2 for i in indices], users_sum['GB hours']*(1000/resources[p]['MaxMemPerNode']), bar_height, label='Node-memory')
    # Set the labels and title
    ax.set_ylabel('User')
    ax.set_xlabel('Resource use (hours)')
    ax.set_title(f'Users resource use - partition {p}')
    ax.set_yticks(indices)
    ax.set_yticklabels(users_sum.index)
    # ax.legend(ncol=2, loc='lower right')
    ax.legend(ncol=2, loc='center')
    fn = 'users-summary.png'
    plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
    plt.savefig(plt_fp)
    plt.close(fig)

    weekly_sum = df_plt_p.groupby('User').resample('W').sum().drop('User', axis=1)
    weekly_sum['CPUxGB hours'] = weekly_sum['GB hours'] * weekly_sum['CPU hours']
    weekly_sum['CPUxTB hours'] = weekly_sum['CPUxGB hours'] * 0.001
    weekly_sum['TB hours'] = weekly_sum['GB hours'] * 0.001
    weekly_count = df_plt_p[['User']].groupby('User').resample('W').count().rename(columns={'User':'NJobs'})
    weekly_sum = weekly_sum.join(weekly_count)

    # Get most active users during recent weeks
    w = args.weeks or 12  # default 12 weeks
    period = pd.Timedelta(w, 'W')
    d_recent = weekly_sum.swaplevel(0, 1).sort_index().loc[ (pd.Timestamp.now() - period).date().isoformat() : ].swaplevel(0, 1)
    n_active_users = len(d_recent)

    plot_max_active_users = min(8, n_active_users)

    for dc in ['CPU hours', 'TB hours', 'CPUxTB hours', 'NJobs', 'Elapsed hours']:
      active_users = d_recent[[dc]].groupby('User').sum().sort_values(dc).index[-plot_max_active_users:]
      plot_max_active_users = min(8, len(active_users))

      set_yaxis_log = False
      if len(active_users) > 1:
        biggest_user = active_users[-1]
        dc_agg = d_recent[dc].groupby('User').sum()
        biggest_user_ratio = dc_agg.loc[active_users[-1]] / dc_agg.loc[active_users[-2]]
        if biggest_user_ratio > 20:
          set_yaxis_log = True

      fig, ax = plt.subplots(figsize=(18, 8))
      bar_width = 0.4 / plot_max_active_users
      for i in range(plot_max_active_users-1, -1, -1):
        u = active_users[i]
        d = d_recent.xs(u, level='User')
        #
        # Line plot:
        # ax.plot(d.index, d[dc], label=u)
        #
        # Bar plot:
        offset = i * bar_width
        bar_positions = np.arange(d.shape[0]) + offset
        ax.bar(bar_positions, d[dc], width=bar_width, label=u)
      if set_yaxis_log:
        ax.set_ylabel(dc + ' log-scale') ; plt.yscale('log')
      else:
        ax.set_ylabel(dc)
      is_bar_chart = False
      for c in ax.get_children():
        # if isinstance(c, matplotlib.container.BarContainer):
        if isinstance(c, matplotlib.patches.Rectangle) and c.get_width() != 1.0:
          is_bar_chart = True
          break
      if is_bar_chart:
        # Fix X-axis labels
        weeks = d_recent.index.get_level_values('Submit').unique()
        ax.set_xticks(np.arange(len(weeks)) + bar_width * (plot_max_active_users / 2 - 0.5))
        ax.set_xticklabels([week.strftime('%Y-%m-%d') for week in weeks], rotation=45)
      plt.legend(title='User', loc='upper left', bbox_to_anchor=(1,1))
      plt.grid(axis='y')
      plt.title(f'Active users, metric={dc}, last {w} weeks - partition {p}')
      fn = f'users-active-{dc}.png'
      plt_fp = os.path.join(plot_dp, '' if p=='*' else p.upper(), fn)
      plt.savefig(plt_fp)
      plt.close(fig)


if args.resources:
  fn_analyse_resources(df)

if args.waits:
  fn_analyse_waiting(df)

if args.users:
  fn_analyse_users(df)

##
