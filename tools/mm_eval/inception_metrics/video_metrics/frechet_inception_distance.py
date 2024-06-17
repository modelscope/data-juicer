# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils
from tools.mm_eval.inception_metrics import distributed
import copy
import math

# fmt: off
#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen, use_image_dataset=True, num_frames=1, subsample_factor: int=1):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    if opts.detector_path is None:
        detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    else:
        detector_url = opts.detector_path
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    if use_image_dataset:
        assert num_frames == 1
        assert subsample_factor == 1
    else:
        opts = copy.deepcopy(opts)
        opts.dataset_kwargs.seq_length = num_frames
        opts.dataset_kwargs.min_spacing = subsample_factor
        opts.dataset_kwargs.max_spacing = subsample_factor

    batch_size = 4

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=0,
        capture_mean_cov=True, max_items=max_real, use_image_dataset=use_image_dataset, div_feature_dim=4).get_mean_cov()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames, subsample_factor=subsample_factor)

    mu_gen, sigma_gen = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs, batch_size=batch_size, rel_lo=0, rel_hi=1,
        capture_mean_cov=True, max_items=num_gen, use_image_dataset=use_image_dataset, div_feature_dim=4, **gen_kwargs).get_mean_cov()

    if distributed.get_rank() != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, e = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member

    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

# ----------------------------------------------------------------------------