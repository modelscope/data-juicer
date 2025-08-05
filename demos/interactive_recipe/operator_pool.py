import oyaml as yaml
import os
import re
import pandas as pd
import json
from collections import OrderedDict
import numpy as np
from loguru import logger

from get_op_info import TYPE_MAPPING

def construct_dj_stats_dict(json_path="./configs/dj_stats_dict.json"):
    dj_stats_dict = {}
    with open(json_path, 'r') as json_file:
        dj_stats_dict = json.load(json_file)
    return dj_stats_dict


DJ_STATS_DICT = construct_dj_stats_dict()

class OperatorArg:
    name = None
    desc = None
    type = None
    v_default = None
    v = None
    v_options = None
    v_min = None
    v_max = None

    def __init__(self, op, state: dict = None):
        self.op = op
        self.name = state.get('name', None)
        if self.name is None:
            raise ValueError("OperatorArg __init__: state['name'] is required.")
        desc = state.get('desc', "no description")
        self.desc = desc
        self.type = state.get('type', None)
        if self.type is None:
            raise ValueError("OperatorArg __init__: state['type'] is required.")
        if self.type not in TYPE_MAPPING:
            raise ValueError(f"OperatorArg __init__: state['type'] = {self.type} is invalid.")
        self.v_default = state.get('default', None)
        if self.v_default is None:
            raise ValueError("OperatorArg __init__: state['default'] is required.")
        self.v_options = state.get('options', None)
        if self.v_type == bool:
            self.v_options = [True, False]
        self.v_min = state.get('min', None)
        self.v_max = state.get('max', None)

        if self.v_min is not None:
            self.v_min = self.v_check(self.v_min, type_only=True)
        if self.v_max is not None:
            self.v_max = self.v_check(self.v_max, type_only=True)
        self.v_default = self.v_check(self.v_default)
        self.v = state.get('v', self.v_default)

    @property
    def state(self):
        return dict(
            name=self.name,
            desc=self.desc,
            type=self.type,
            default=self.v_default,
            v=self.v,
            options=self.v_options,
            min=self.v_min,
            max=self.v_max,
        )

    @property
    def v_type(self):
        return TYPE_MAPPING[self.type]

    def v_check(self, v, type_only=False, element_check=False):
        error = ValueError(
            f"OperatorArg v_check: type mismatch when check {self.name}={v}, \
            expected {self.v_type} but got {type(v)}."
        )
        try:
            v = self.v_type(v)
        except:
            raise error
        if type_only:
            return v
        if self.v_options is not None and v not in self.v_options:
            raise ValueError(f"OperatorArg v_check: {self.name}={v} is not in options {self.v_options}.")
        if self.v_min is not None and v < self.v_min:
            logger.warning(f"OperatorArg v_check: {self.name}={v} is less than v_min={self.v_min}.")
            return self.v_min
        if self.v_max is not None and v > self.v_max:
            logger.warning(f"OperatorArg v_check: {self.name}={v} is larger than v_max={self.v_max}.")
            return self.v_max
        # TODO: silence cross arg value check temporarily, active it with better implementation
        # if self.name.startswith('min_'):
        #     query_key = f'max_{self.name[4:]}'
        #     if query_key in self.op.args and v > self.op.args[query_key].v:
        #         raise ValueError(f"OperatorArg v_check: \
        #                            {self.name}={v} is greater than {query_key}={self.op.args[query_key].v}")
        # if self.name.startswith('max_'):
        #     query_key = f'min_{self.name[4:]}'
        #     if query_key in self.op.args and v < self.op.args[query_key].v:
        #         raise ValueError(f"OperatorArg v_check: \
        #                            {self.name}={v} is less than {query_key}={self.op.args[query_key].v}")
        return v

    @property
    def stats_apply(self):
        # if op stats apply to this argument
        return self.op.dj_stats_key is not None and (self.name.startswith('min_') or self.name.startswith('max_'))

    @property
    def quantiles(self):
        return self.op.quantiles

    def update_with_stats(self, stats):
        if not self.stats_apply:
            return
        self.v_min = self.v_type(stats['min'])
        self.v_max = self.v_type(stats['max'])
        if self.v < self.v_min:
            self.v = self.v_min
        elif self.v > self.v_max:
            self.v = self.v_max

    def _p2v(self, p):
        # percentage to value
        if self.quantiles is None:
            raise ValueError("OperatorArg _p2v: quantiles is required.")
        return self.v_type(self.quantiles[int(p)])

    def _v2p(self, v):
        # value to percentage
        # dichotomy
        if self.quantiles is None:
            raise ValueError("OperatorArg _v2q: quantiles is required.")
        l, r = 0, 100
        while l < r - 1:
            m = (l + r) // 2
            if self.quantiles[m] < v:
                l = m + 1
            else:
                r = m
        return l

    def set_v(self, v):
        v = self.v_check(v)
        self.v = v
        self.save()

    def set_p(self, p):
        if 0 <= p <= 1:
            p = int(p * 100)
        if not isinstance(p, int) or p < 0 or p > 100:
            raise ValueError("OperatorArg set_p: p is invalid.")
        v = self._p2v(p)
        self.set_v(v)

    def set_k(self, k, stats):
        if not self.stats_apply:
            return
        mean, std = stats.get("mean", None), stats.get("std", None)
        if mean is None or std is None:
            raise ValueError("OperatorArg set_k: mean and std are required. Please run update_with_stats first.")
        if self.name.startswith('min_'):
            self.set_v(max(self.v_min, self.v_type(mean - k * std)))
        elif self.name.startswith('max_'):
            self.set_v(min(self.v_max, self.v_type(mean + k * std)))

    def save(self):
        self.op.save()


class Operator:
    name = None
    desc = None
    enabled = None
    args = None
    stats = None

    def __init__(self, pool, state: dict = None):
        self.pool = pool
        self.name = state.get('name', None)
        self.desc = state.get('desc', "no description")
        args = state.get('args', {})
        logger.info(f"Operator: __init__: {self.name} args: {args}")
        self.enabled = state.get('enabled', True)
        self.args = OrderedDict()
        for arg_name, arg_state in args.items():
            arg_state.update(name=arg_name)
            self.args[arg_name] = OperatorArg(self, state=arg_state)
        self.stats = state.get('stats', {})

    @property
    def quantiles(self):
        return self.stats.get('quantiles', None)

    @property
    def dj_stats_key(self):
        return DJ_STATS_DICT.get(self.name, None)

    @property
    def state(self):
        args = {}
        for arg_name in self.args:
            args[arg_name] = self.args[arg_name].state
        return dict(
            name=self.name,
            desc=self.desc,
            enabled=self.enabled,
            args=args,
            stats=self.stats,
        )

    def update_with_stats(self, stats):
        self.stats.update(stats)
        for arg_name in self.args:
            self.args[arg_name].update_with_stats(stats)

    def disable(self):
        self.enabled = False
        self.save()

    def enable(self):
        self.enabled = True
        self.save()

    def set_k(self, k):
        for arg_name in self.args:
            self.args[arg_name].set_k(k, self.stats)

    def save(self):
        self.pool.save()


class OperatorPool:
    config_path = None
    default_ops = None
    pool = None

    def __init__(self, config_path=None, default_ops=None):
        if config_path is None and default_ops is None:
            config_path = os.path.join(os.path.dirname(__file__), "./configs/all_op_info.yaml")
        if default_ops is None:
            with open(config_path, "r") as f:
                self.default_ops = yaml.safe_load(f)
        else:
            self.default_ops = default_ops
        self.pool = OrderedDict()
        for op_name, arg_state in self.default_ops.items():
            arg_state.update(name=op_name)
            self.pool[op_name] = Operator(self, state=arg_state)

    def __getitem__(self, i):
        if isinstance(i, int):
            return list(self.pool.values())[i]
        return self.pool.get(i, None)

    def __contains__(self, key):
        return key in self.pool

    def __iter__(self):
        return iter(self.pool)

    @property
    def state(self):
        s = {}
        for op_name in self.pool:
            s[op_name] = self.pool[op_name].state
        return s

    def export_config(self, project_name, dataset_path, nproc, export_path, config_path=None):
        if config_path is None:
            config_path = "./configs/demo.yaml"
        config = {
            "project_name": project_name,
            "dataset_path": dataset_path,
            "np": nproc,
            "export_path": export_path,
        }
        process = []
        for op_name, op in self.pool.items():
            if op.enabled:
                args = {}
                for arg_name, arg in op.args.items():
                    args[arg_name] = arg.v
                process.append({op_name: args})
        config["process"] = process
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return config_path

    def act(self, op_name, action_type, *args, **kwargs):
        """
        :param op_name: target operator name
        :param action_type: one of ['enable', 'disable', 'set_arg']
            'enable': enable an operator if it is disabled, else nothing to do.
            'disable': disable an operator if it is enabled, else nothing to do.
            'set_arg': set operator argument
        :param args:
        :param kwargs:
            if action_type=='set_arg', kwarg['arg_name'] is required
        :return:
        """
        if op_name not in self.pool:
            raise ValueError(f"OperatorPool: invalid act: {op_name} is not in pool.")
        if action_type == 'enable':
            self.pool[op_name].enable()
        elif action_type == 'disable':
            self.pool[op_name].disable()
        elif action_type == 'set_arg':
            if 'arg_name' not in kwargs:
                if "k" not in kwargs:
                    raise ValueError("OperatorPool: invalid action: \
                                      parameter 'arg_name' or 'k' is required for action_type='set_arg'.")
                # set mean pm k * std
                self.pool[op_name].set_k(kwargs["k"])
            elif 'v' in kwargs:
                v = kwargs['v']
                try:
                    self.pool[op_name].args[kwargs["arg_name"]].set_v(v)
                except Exception as e:
                    logger.error(e)
            elif 'p' in kwargs:
                p = kwargs['p']
                try:
                    self.pool[op_name].args[kwargs["arg_name"]].set_p(p)
                except Exception as e:
                    logger.error(e)
            else:
                raise ValueError("OperatorPool: invalid action: \
                                  parameter 'v' or 'p' is required for action_type='set_arg'.")
        else:
            raise ValueError(f"OperatorPool: invalid action_type: {action_type}, \
                                           should be one of ['enable', 'disable', 'set_arg'].")

    def save(self, path="./save/op_pool_state.yaml"):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.state, f)
