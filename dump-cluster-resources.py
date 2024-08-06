import subprocess
import re
import os
import datetime
import json

from utils import get_cluster_name

def expand_node_ranges(node_range):
    nodes = []
    ranges = node_range.split(',')
    for r in ranges:
        if '[' in r and ']' in r:
            prefix = r.split('[')[0]
            range_part = r.split('[')[1].split(']')[0]
            for part in range_part.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    nodes.extend([f"{prefix}{i}" for i in range(int(start), int(end)+1)])
                else:
                    nodes.append(f"{prefix}{part}")
        else:
            nodes.append(r)
    return nodes

def get_node_info():
    node_info = subprocess.check_output(['scontrol', 'show', 'nodes'], text=True)
    nodes = {}
    current_node = None
    
    for line in node_info.splitlines():
        if line.startswith('NodeName='):
            match = re.search(r'NodeName=(\S+)', line)
            if match:
                current_node = match.group(1)
                nodes[current_node] = {'CPUs': 0, 'Memory': 0}
        
        if current_node:
            if 'CPUTot=' in line:
                match = re.search(r'CPUTot=(\d+)', line)
                if match:
                    nodes[current_node]['CPUs'] = int(match.group(1))
            if 'RealMemory=' in line:
                match = re.search(r'RealMemory=(\d+)', line)
                if match:
                    nodes[current_node]['Memory'] = int(match.group(1))
            if 'Gres=' in line and 'Gres=(null)' not in line:
                match = re.search(r"Gres=(gpu:[^\s]+)", line)
                gpu_info_items = match.group(1).split(':')
                # Until I see real examples otherwise, assume syntax is:
                # gpu:[model]:[count]
                model = gpu_info_items[1]
                count = int(gpu_info_items[2])
                nodes[current_node]['GPUs'] = [model, count]
    
    return nodes

def get_partition_info():
    partition_info = subprocess.check_output(['scontrol', 'show', 'partition'], text=True)
    partitions = {}
    current_partition = None
    
    for line in partition_info.splitlines():
        if line.startswith('PartitionName='):
            match = re.search(r'PartitionName=(\S+)', line)
            if match:
                current_partition = match.group(1)
                partitions[current_partition] = {'Nodes': [], 'MaxCPUsPerNode': 0, 'MaxMemPerNode': 0}
        
        if current_partition:
            if re.search(r'\bNodes=\S+', line):
                match = re.search(r'\bNodes=(\S+)', line)
                if match:
                    node_range = match.group(1)
                    nodes = expand_node_ranges(node_range)
                    partitions[current_partition]['Nodes'].extend(nodes)
            if 'MaxCPUsPerNode=' in line:
                match = re.search(r'MaxCPUsPerNode=(\d+)', line)
                if match:
                    partitions[current_partition]['MaxCPUsPerNode'] = int(match.group(1))
            if 'MaxMemPerNode=' in line:
                match = re.search(r'MaxMemPerNode=(\d+)', line)
                if match:
                    partitions[current_partition]['MaxMemPerNode'] = int(match.group(1))
    
    return partitions

def prune_nodes(partitions, nodes):
    node2parts = {}
    for p in partitions:
        for n in partitions[p]['Nodes']:
            if n not in node2parts:
                node2parts[n] = [p]
            else:
                node2parts[n].append(p)

    for part in partitions:
        partitions[part]['Nodes pruned'] = list(partitions[part]['Nodes'])
    for node, part_list in node2parts.items():
        if len(part_list) > 1:
            # This node appears in multiple partitions. 
            # Check if one/some of them are "meta" partitions, e.g. a generic "gpu" partition

            node_cpus = nodes[node]['CPUs']
            node_memory = nodes[node]['Memory']
            assigned_cpus = sum(partitions[part]['MaxCPUsPerNode'] for part in part_list)
            assigned_memory = sum(partitions[part]['MaxMemPerNode'] for part in part_list)

            if assigned_cpus > node_cpus or assigned_memory > node_memory:
                # Node resources are oversubscribed, implies one/some of partitions are meta.
                # Only maintain relationship to partition with fewest nodes.
                minp = min(part_list, key=lambda part: len(partitions[part]['Nodes']))
                for part in part_list:
                    if part != minp:
                        partitions[part]['Nodes pruned'].remove(node)
            else:
                # Record the % allocated to each partition
                for part in part_list:
                    pct_cpus = partitions[part]['MaxCPUsPerNode'] / node_cpus
                    pct_mem = partitions[part]['MaxMemPerNode'] / node_memory
                    if 'Node CPU %' not in partitions[part]:
                        partitions[part]['Node CPU %'] = {node:pct_cpus}
                    else:
                        partitions[part]['Node CPU %'][node] = pct_cpus
                    if 'Node mem %' not in partitions[part]:
                        partitions[part]['Node mem %'] = {node:pct_mem}
                    else:
                        partitions[part]['Node mem %'][node] = pct_mem
    for part in list(partitions.keys()):
        if len(partitions[part]['Nodes pruned']) == 0:
            del partitions[part]
        else:
            partitions[part]['Nodes'] = partitions[part]['Nodes pruned']
            del partitions[part]['Nodes pruned']
    return partitions

d = os.path.join('Dumps', get_cluster_name(), 'resources')
if not os.path.isdir(d):
    os.makedirs(d)
dt = datetime.date.today()
nodes = get_node_info()
fp = os.path.join('Dumps', get_cluster_name(), 'resources', f'nodes-{str(dt)}.json')
if not os.path.isfile(fp):
    try:
        with open(fp, 'w') as f:
            json.dump(nodes, f)
        print("Slurm node info written to: "+ fp)
    except Exception:
        if os.path.isfile(fp):
            os.remove(fp)

partitions = get_partition_info()
partitions = prune_nodes(partitions, nodes)
fp = os.path.join('Dumps', get_cluster_name(), 'resources', f'partitions-{str(dt)}.json')
if not os.path.isfile(fp):
    try:
        with open(fp, 'w') as f:
            json.dump(partitions, f)
        print("Slurm partition info written to: "+ fp)
    except Exception:
        if os.path.isfile(fp):
            os.remove(fp)
