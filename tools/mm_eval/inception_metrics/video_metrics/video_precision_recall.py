# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Precision/Recall (PR) from the paper "Improved Precision and Recall
Metric for Assessing Generative Models". Matches the original implementation
by Kynkaanniemi et al. at
https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py"""

import copy
import torch
from . import metric_utils
from tools.mm_eval.inception_metrics import distributed

#----------------------------------------------------------------------------

def compute_distances(row_features, col_features, world_size, rank, col_batch_size):
    assert 0 <= rank < world_size
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // world_size + 1) * world_size
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: world_size]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(world_size):
            dist_broadcast = dist_batch.clone()
            if world_size > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_pr(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size, num_frames: int, subsample_factor: int=1, use_image_dataset=False):
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
        batch_size=batch_size, use_image_dataset=use_image_dataset).get_all_torch()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts

    gen_features = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen, temporal_detector=True,
        batch_size=batch_size, use_image_dataset=use_image_dataset).get_all_torch()

    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, world_size=world_size, rank=rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if rank == 0 else None)
        kth = torch.cat(kth) if rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, world_size=world_size, rank=rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if rank == 0 else 'nan')
    return results['precision'], results['recall']

#----------------------------------------------------------------------------