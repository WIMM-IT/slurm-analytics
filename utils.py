import subprocess

def get_cluster_name():
    config = subprocess.check_output(['scontrol', 'show', 'config']).decode('ascii')
    for line in config.splitlines():
        if line.startswith("ClusterName"):
            return line.split("=")[1].strip()
