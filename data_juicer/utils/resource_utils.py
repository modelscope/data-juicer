import subprocess

import psutil
from loguru import logger

NVSMI_REPORT = True


def query_cuda_info(query_key):
    global NVSMI_REPORT
    # get cuda info using "nvidia-smi" command in MB
    try:
        nvidia_smi_output = subprocess.check_output([
            'nvidia-smi', f'--query-gpu={query_key}',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')
    except Exception as e:
        if 'non-zero exit status 2' in str(e):
            err_msg = f'The specified query_key [{query_key}] might not be ' \
                      f'supported by command nvidia-smi. Please check and ' \
                      f'retry!'
        elif 'No such file or directory' in str(e):
            err_msg = 'Command nvidia-smi is not found. There might be no ' \
                      'GPUs on this machine.'
        else:
            err_msg = str(e)
        if NVSMI_REPORT:
            logger.warning(err_msg)
            NVSMI_REPORT = False
        return None
    cuda_info_list = []
    for line in nvidia_smi_output.strip().split('\n'):
        cuda_info_list.append(int(line))
    return cuda_info_list


def get_cpu_count():
    return psutil.cpu_count()


def get_cpu_utilization():
    return psutil.cpu_percent()


def query_mem_info(query_key):
    mem = psutil.virtual_memory()
    if query_key not in mem._fields:
        logger.warning(f'No such query key [{query_key}] for memory info. '
                       f'Should be one of {mem._fields}')
        return None
    val = round(mem.__getattribute__(query_key) / (2**20), 2)  # in MB
    return val
