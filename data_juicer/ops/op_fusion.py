from typing import List

import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, InterVars
from data_juicer.utils.registry import Registry

from .base_op import Filter

# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# audios
LOADED_AUDIOS = Registry(InterVars.loaded_audios)

# videos
LOADED_VIDEOS = Registry(InterVars.loaded_videos)
INTER_SAMPLED_FRAMES = Registry(InterVars.sampled_frames)

# all
ALL_INTER_VARS = [
    INTER_LINES, INTER_WORDS, LOADED_IMAGES, LOADED_VIDEOS,
    INTER_SAMPLED_FRAMES
]

# supported fusion strategies
FUSION_STRATEGIES = {'greedy', 'probe'}


def fuse_operators(ops, probe_res=None):
    """
    Fuse the input ops list and return the fused ops list.

    :param ops: the corresponding list of op objects.
    :param probe_res: the probed speed for each OP from Monitor.
    :return: a list of fused op objects.
    """
    if probe_res is None:
        probe_res = [None for _ in range(len(ops))]
    # detect filter groups and try to fuse them
    fused_ops = []
    filter_group = []
    in_group = False
    for op, op_probe in zip(ops, probe_res):
        if isinstance(op, Filter):
            if not in_group:
                in_group = True
            filter_group.append((op, op_probe))
        elif in_group:
            # got a filter group, try to fuse them
            fused_group = fuse_filter_group(filter_group)
            fused_ops.extend(fused_group)
            filter_group = []
            in_group = False
            # and add the current non-filter op into fused_ops
            fused_ops.append(op)
        else:  # not a filter and not in a filter group, skip
            fused_ops.append(op)
    if in_group and len(filter_group) > 0:
        # the final filter group, try to fuse them
        fused_group = fuse_filter_group(filter_group)
        fused_ops.extend(fused_group)
    return fused_ops


def fuse_filter_group(original_filter_group):
    """
    Fuse single filter group and return the fused filter group.

    :param original_filter_group: the original filter group, including op
        definitions and objects.
    :return: the fused definitions and objects of the input filter group.
    """
    fused_group = []
    group_speed = []
    all_intermediate_vars = ALL_INTER_VARS
    all_fused_filters = {
        inter_vars: []
        for inter_vars in all_intermediate_vars
    }
    # group these filters by their intermediate vars
    for op, probe_res in original_filter_group:
        op_name = op._name
        for inter_vars in all_intermediate_vars:
            if op_name in inter_vars.modules:
                all_fused_filters[inter_vars].append((op, probe_res))
                break
        else:
            # first apply other filters to decrease the number of samples, so
            # we add them into the fused_group list directly
            fused_group.append(op)
            group_speed.append(probe_res['speed'] if probe_res else 0)

    # try to fuse ops for each type of intermediate vars
    for inter_vars in all_intermediate_vars:
        inter_vars_filter = all_fused_filters[inter_vars]
        if len(inter_vars_filter) == 0:
            # no ops include this type of intermediate var
            pass
        elif len(inter_vars_filter) > 1:
            # more than 1 ops share the same intermediate var, try to fuse them
            ops, probe_res_list = zip(*inter_vars_filter)
            # new definition: new name and a definition list of fused op list
            fused_filter_name = 'OpFusion:(%s)' % ','.join(
                [op._name for op in ops])
            logger.info(f'Ops are fused into one op '
                        f'{fused_filter_name}.')
            # use these ops to create a FusedFilter object, and add the fused
            # definition and op into the fused group
            fused_filter = FusedFilter(fused_filter_name, ops)
            fused_filter._op_cfg = {
                fused_filter_name: [op._op_cfg for op in ops]
            }
            fused_filter_speed = sum([
                1.0 / probe_res['speed'] for probe_res in probe_res_list
                if probe_res
            ])
            if fused_filter_speed > 0:
                fused_filter_speed = 1.0 / fused_filter_speed
            fused_group.append(fused_filter)
            group_speed.append(fused_filter_speed)
        else:
            # only 1 op for this type of intermediate var, add it to the fused
            # group directly without fusion
            fused_group.append(inter_vars_filter[0][0])
            probe_res = inter_vars_filter[0][1]
            group_speed.append(probe_res['speed'] if probe_res else 0)

    # reorder according to the probed speed results in group_speed
    # 'greedy': all speed data in group_speed will be 0, which will keep the
    #   current order of fused group
    # 'probe': OPs in fused group will be reordered according to the speed data
    #   in group_speed in descending order
    fused_group = [
        op for op, _ in sorted(
            zip(fused_group, group_speed), key=lambda it: it[1], reverse=True)
    ]

    return fused_group


class FusedFilter(Filter):
    """A fused operator for filters."""

    _batched_op = True

    def __init__(self, name: str, fused_filters: List):
        """
        Initialization method.

        :param fused_filters: a list of filters to be fused.
        """
        super().__init__()
        self._name = name
        self.fused_filters = fused_filters
        # set accelerator to 'cuda' if there exists any ops whose accelerator
        # is 'cuda'
        accelerator_methods = set(
            [op.accelerator for op in self.fused_filters])
        if 'cuda' in accelerator_methods:
            self.accelerator = 'cuda'

        # update num_proc with the min num_proc of all fusible filters
        self.num_proc = min([op.runtime_np() for op in self.fused_filters])

    def compute_stats_batched(self, samples, rank=None):
        import av

        # context for the intermediate vars
        num_samples = len(samples[Fields.stats])
        samples[Fields.context] = [{} for _ in range(num_samples)]
        for op in self.fused_filters:
            # open the context for these fused ops
            if op.accelerator == 'cuda':
                samples = op.compute_stats_batched(samples,
                                                   rank=rank,
                                                   context=True)
            else:
                samples = op.compute_stats_batched(samples, context=True)
        # clean up the contexts after processing
        # check if there are containers that need to be closed
        for ctx in samples[Fields.context]:
            for context_key in ctx:
                if isinstance(ctx[context_key], av.container.InputContainer):
                    ctx[context_key].streams.video[0].close()
                    ctx[context_key].close()
        _ = samples.pop(Fields.context)
        return samples

    def process_batched(self, samples):
        # Only return True when all filters return True
        res = None
        for op in self.fused_filters:
            this_res = np.array(list(op.process_batched(samples)))
            if res is not None:
                res = np.logical_and(res, this_res)
            else:
                res = this_res
        return res
