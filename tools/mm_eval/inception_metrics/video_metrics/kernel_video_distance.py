# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kernel Inception Distance (KID) from the paper "Demystifying MMD
GANs". Matches the original implementation by Binkowski et al. at
https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py"""

import copy
import numpy as np
from . import metric_utils
from tools.mm_eval.inception_metrics import distributed

# fmt: off
#----------------------------------------------------------------------------

def compute_kid(opts, max_real, num_gen, num_subsets, max_subset_size, num_frames: int, subsample_factor: int=1, use_image_dataset=False):
    # Perfectly reproduced torchscript version of the I3D model, trained on Kinetics-400, used here:
    # https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
    # Note that the weights on tf.hub (used in the script above) differ from the original released weights
    if opts.detector_path is None:
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    else:
        detector_url = opts.detector_path
    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    opts = copy.deepcopy(opts)
    opts.dataset_kwargs.seq_length = num_frames
    opts.dataset_kwargs.min_spacing = subsample_factor
    opts.dataset_kwargs.max_spacing = subsample_factor

    opts.gen_dataset_kwargs.seq_length = num_frames
    opts.gen_dataset_kwargs.min_spacing = subsample_factor
    opts.gen_dataset_kwargs.max_spacing = subsample_factor

    batch_size = max(1, 64 // num_frames)

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real, temporal_detector=True,
        batch_size=batch_size, use_image_dataset=use_image_dataset).get_all()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames, subsample_factor=subsample_factor)

    gen_features = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen, temporal_detector=True,
        batch_size=batch_size, use_image_dataset=use_image_dataset).get_all()

    if distributed.get_rank() != 0:
        return float('nan')

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

# ----------------------------------------------------------------------------