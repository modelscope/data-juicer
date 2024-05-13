# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/universome/stylegan-v/blob/master/src/metrics/metric_main.py

import os
import time
import json
import torch
import numpy as np
from tools.mm_eval.inception_metrics import distributed
from tools.mm_eval.inception_metrics.util import EasyDict, format_time

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import inception_score
from . import precision_recall
from . import frechet_video_distance
from . import kernel_video_distance
from . import video_inception_score
from . import video_precision_recall


# fmt: off
#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

def is_power_of_two(n: int) -> bool:
    return (n & (n-1) == 0) and n != 0

#----------------------------------------------------------------------------

def calc_metric(metric, num_runs: int=1, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)
    world_size = distributed.get_world_size()

    # Calculate.
    start_time = time.time()
    all_runs_results = [_metric_dict[metric](opts) for _ in range(num_runs)]
    total_time = time.time() - start_time

    # Broadcast results.
    for results in all_runs_results:
        for key, value in list(results.items()):
            if world_size > 1:
                value = torch.as_tensor(value, dtype=torch.float64, device="cuda")
                torch.distributed.broadcast(tensor=value, src=0)
                value = float(value.cpu())
            results[key] = value

    if num_runs > 1:
        results = {f'{key}_run{i+1:02d}': value for i, results in enumerate(all_runs_results) for key, value in results.items()}
        for key, value in all_runs_results[0].items():
            all_runs_values = [r[key] for r in all_runs_results]
            results[f'{key}_mean'] = np.mean(all_runs_values)
            results[f'{key}_std'] = np.std(all_runs_values)
    else:
        results = all_runs_results[0]

    # Decorate with metadata.
    return EasyDict(
        results         = EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = format_time(total_time),
        world_size      = world_size,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------

@register_metric
def fid50k(opts):
    '''
        Compute Frechet Inception Distance (FID) of frames, sample 50000 frames from fake dataset at most
    '''
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    '''
        Compute Kernel Inception Distance (KID) of frames, sample 50000 frames from fake dataset at most, split features to 100 subset to compute KIDs and return the mean.
    '''
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def is50k(opts):
    '''
        Compute Inception Score(IS) of frames, sample 50000 frames from fake dataset at most, split features to 10 subset to compute ISs and return the mean and std.
    '''
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

@register_metric
def pr50k_3n(opts):
    '''
        Compute Precision/Recall (PR) of frames, sample 50000 frames from fake dataset at most, with the 4th (3+1) nearest features to estimate the distributions
    '''
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k_3n_precision=precision, pr50k_3n_recall=recall)

@register_metric
def fvd2048_16f(opts):
    '''
        Compute Frechet Video Distance (FVD), sample 2048 times in dataset, 16 adjacent frames each time.
    '''
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=16)
    return dict(fvd2048_16f=fvd)

@register_metric
def fvd2048_128f(opts):
    '''
        Compute Frechet Video Distance (FVD), sample 2048 times in dataset, 128 adjacent frames each time.
    '''
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=128)
    return dict(fvd2048_128f=fvd)

@register_metric
def fvd2048_128f_subsample8f(opts):
    '''
        Compute Frechet Video Distance (FVD), sample 2048 times in dataset, 16 frames each time, sample 1 frame every adjacent 8 frames.
    '''
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=16, subsample_factor=8)
    return dict(fvd2048_128f_subsample8f=fvd)

@register_metric
def kvd2048_16f(opts):
    '''
        Compute Kernel Video Distance (KVD), sample 2048 times in dataset, 16 adjacent frames each time, split features to 100 subset to compute KVDs and return the mean.
    '''
    kid = kernel_video_distance.compute_kid(opts, max_real=2048, num_gen=2048, num_frames=16, num_subsets=100, max_subset_size=1000)
    return dict(kvd2048_16f=kid)

@register_metric
def isv2048_ucf(opts):
    '''
        Compute Inception Score of Videos (ISV), sample 2048 times in dataset, 16 adjacent frames each time, split features to 10 subset to compute ISs and return the mean and std.
    '''
    mean, std = video_inception_score.compute_isv(opts, num_gen=2048, num_splits=10, backbone='c3d_ucf101')
    return dict(isv2048_ucf_mean=mean, isv2048_ucf_std=std)


@register_metric
def prv2048_3n_16f(opts):
    '''
        Compute Precision/Recall of Videos (PRV), sample 2048 times in dataset, 16 adjacent frames each time, with the 4th (3+1) nearest features to estimate the distributions
    '''
    precision, recall = video_precision_recall.compute_pr(opts, max_real=2048, num_gen=2048, nhood_size=3, row_batch_size=10000, col_batch_size=10000, num_frames=16)
    return dict(prv2048_3n_16f_precision=precision, prv2048_3n_16f_recall=recall)
