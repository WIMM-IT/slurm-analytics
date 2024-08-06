#!/usr/bin/python3

import os
import datetime as _dt
import time
import subprocess

from utils import get_cluster_name
cluster_id = get_cluster_name()
period = _dt.timedelta(days=7)


# Having 'sacct' dump everything is very expensive and certain
# to disrupt users' experience. Better to dump in small chunks.
# This requires logic to detect what chunks have already been dumped, 
# and calculate the next chunk to dump.



# Important to also know resources in each partition:
filename = 'dump-cluster-resources.py'
with open(filename) as file:
    exec(file.read())


# Confirmed working
def get_dates_from_filepath(filepath):
    filename = os.path.basename(filepath)
    dates_str = '-'.join(filename.split('-')[2:])
    dates_str = dates_str[:-len(".csv")]
    start_str, end_str = dates_str.split("_")
    start_date = _dt.datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = _dt.datetime.strptime(end_str, "%Y-%m-%d").date()
    return start_date, end_date


# Confirmed working
def get_most_recent_file_date(directory):
    # Finds the most recent date from the filenames in the directory
    if not os.path.isdir(directory):
        return None
    most_recent_date = None
    for filename in os.listdir(directory):
        file_date = get_dates_from_filepath(filename)
        if file_date and (not most_recent_date or file_date > most_recent_date):
            most_recent_date = file_date
    return most_recent_date


# Confirmed working
def get_oldest_file_date(directory):
    # Finds the oldest date from the filenames in the directory
    if not os.path.isdir(directory):
        return None
    oldest_date = None
    for filename in os.listdir(directory):
        file_date = get_dates_from_filepath(filename)
        if file_date and (not oldest_date or file_date < oldest_date):
            oldest_date = file_date
    return oldest_date


# Confirmed working
def delete_outdated_files(directory):
    # Deletes files where the end date in the filename is after the file's last modification date, 
    # but only if file isn't recent (younger than 1 hour).
    if not os.path.isdir(directory):
        return
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_date = get_dates_from_filepath(filename)
        mod_ts = os.path.getmtime(filepath)
        age_ts = time.time() - mod_ts
        age_hours = age_ts / 3600
        if age_hours < 1.0:
            continue
        end_dt = _dt.datetime.combine(file_date[1], _dt.time(0))
        end_ts = time.mktime(end_dt.timetuple())
        if file_date and mod_ts < end_ts:
            print("Delete outdated:", filepath)
            os.remove(filepath)


# Confirmed working
def get_next_range(directory):
    last = get_most_recent_file_date(directory)
    if last is None:
        end = _dt.date.today()
        start = end - period
    else:
        td = last[1] - last[0]
        if td != period:
            # Not a full week, so fetch just the next few days to complete week
            start = last[1]
            end = last[0] + period
        else:
            start = last[1]
            end = start + period
    return start, end


# Confirmed working
def get_older_range(directory):
    # Find oldest file, and calculate the week preceding it.
    first = get_oldest_file_date(directory)
    if first is None:
        end = _dt.date.today()
        start = end - period
    else:
        # Check if first file is empty, which means no more historic data
        fn = 'sacct-output-'+str(first[0])+'_'+str(first[1])+'.csv'
        fp = os.path.join(directory, fn)
        with open(fp, "r") as f:
            nlines = sum(1 for _ in f)
        if nlines == 1:
            # Just header
            return None

        td = first[1] - first[0]
        if td != period:
            # Not a full week, so fetch just the preceding few days to complete week
            start = first[1] - period
            end = first[0]
        else:
            start = first[0] - period
            end = first[0]
    return start, end


def sacct_get_date_range(start, end):
  fields = 'JobID,User,Submit,Timelimit,State,Partition'  # basic job info
  fields += ',Start,End,Elapsed'  # execution
  fields += ',NCPUS,TotalCPU'  # CPU efficiency
  fields += ',ReqMem,MaxRSS'  # memory efficiency
  fields += ',AllocNodes'  # For user total use
  fields += ',NodeList'  # To help map job to specific partition
  fields += ',Reason'  # To see dependency, important for analysing queue wait time
  fields += ',Priority'  # Trying to figure out why some jobs had big wait times
  cmd = ['sacct', '--units=M', '-p', '--delimiter=|', '--units=M', '--allusers', "-S", start, "-E", end, "-o", fields]
  #cmd.insert(1, '-v')  # verbose

  print(' '.join(cmd))
  # sacct = subprocess.check_output(cmd).decode("UTF8")
  sacct = subprocess.check_output(cmd).decode("ascii")
  return sacct


def sacct_dump_date_range(start, end, directory):
  start = str(start)
  end = str(end)

  cache_fp = os.path.join(directory, 'sacct-output-'+start+'_'+end+'.csv')
  if os.path.isfile(cache_fp):
    print("already have sacct output for date range {} -> {}".format(start, end))
    with open(cache_fp, 'r') as F:
      return F.read()

  sacct = sacct_get_date_range(start, end)

  if not os.path.isdir(directory):
    os.makedirs(directory)
  with open(cache_fp, 'w') as F:
    F.write(sacct)
  print("sacct output written to: " + cache_fp)


output_directory = os.path.join("Dumps", cluster_id, "sacct")
delete_outdated_files(output_directory)
fetch_range = get_next_range(output_directory)
if fetch_range[0] > _dt.date.today():
    fetch_range = get_older_range(output_directory)
    if fetch_range is None:
        print("No more old data to get from Slurm, you have it all!")
        quit()
print("- fetch_range =", fetch_range, type(fetch_range))
sacct_dump_date_range(fetch_range[0], fetch_range[1], output_directory)
