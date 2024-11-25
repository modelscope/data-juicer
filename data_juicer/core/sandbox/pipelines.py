import os
from typing import List

import wandb
import yaml
from jsonargparse import Namespace as JsonNamespace
from jsonargparse import namespace_to_dict

from data_juicer.config import merge_config
from data_juicer.core.sandbox.hooks import register_hook


class SandBoxWatcher:
    """
    Basic Watcher class to manage interested results, and manage the experiment
    within the sandbox based on WandB UI and it's utilities.
    """

    def __init__(self, sandbox_cfg):
        """
        Initialize the watcher with a reference to an executor instance.
        """

        # the web-ui and experiment versioning is based on WandB
        project_name = sandbox_cfg.project_name
        experiment_name = sandbox_cfg.experiment_name
        hpo_config = sandbox_cfg.hpo_config
        self.sandbox_cfg = sandbox_cfg
        if not os.path.exists(self.sandbox_cfg.work_dir):
            os.makedirs(self.sandbox_cfg.work_dir, exist_ok=True)

        self.wandb_run = wandb.init(project=project_name, name=experiment_name)
        if (hpo_config is not None and 'metric' in hpo_config
                and 'name' in hpo_config['metric']):
            self.object_name_in_hpo = hpo_config['metric']['name']
        else:
            self.object_name_in_hpo = None
        self.logged_res = {}

    def query(self, meta_name: str):
        """
        Query the result from the logged_res.
        """
        return self.logged_res.get(meta_name)

    def watch(self, res, meta_name: str = ''):
        """
        Flatten the result in dot structure and log it into WandB.
        """
        if isinstance(res, dict):
            for key, value in res.items():
                # getting the left nodes of the given res dictionary.
                if isinstance(value, dict):
                    self.watch(value, f'{meta_name}.{key}')
                else:
                    self.logged_res[f'{meta_name}.{key}'] = value
                    if self.object_name_in_hpo == f'{meta_name}.{key}':
                        # Ensuring float results for HPO experiments
                        value = float(value)
                    self.wandb_run.log({f'{meta_name}.{key}': value})
        else:
            self.logged_res[meta_name] = res
            if meta_name == self.object_name_in_hpo:
                res = float(res)
            self.wandb_run.log({meta_name: res})

    def setup_sweep(self, hpo_config: dict = None, project_name: str = None):
        """
        Setup and start a new WandB sweep.
        """
        if hpo_config is None:
            hpo_config = self.sandbox_cfg.hpo_config
        if project_name is None:
            project_name = self.sandbox_cfg.project_name
        sweep_id = wandb.sweep(sweep=hpo_config, project=project_name)
        if (hpo_config is not None and 'metric' in hpo_config
                and 'name' in hpo_config['metric']):
            self.object_name_in_hpo = hpo_config['metric']['name']
        return sweep_id

    def watch_cfgs(self, cfgs: List[tuple] = None):
        """
        Watch the configuration of the experiment.
        """
        merged_cfgs = {}
        if cfgs is not None:
            for cfg, cfg_prefix in cfgs:
                # skip empty configs
                if cfg is None:
                    continue
                if isinstance(cfg, JsonNamespace):
                    converged_cfg = namespace_to_dict(cfg)
                elif isinstance(cfg, dict):
                    converged_cfg = cfg
                else:
                    raise ValueError(
                        f'Expected dict or JsonNamespace, got {type(cfg)}')
                for key, val in converged_cfg.items():
                    merged_cfgs[f'{cfg_prefix}.{key}'] = val
        else:
            merged_cfgs = namespace_to_dict(self.sandbox_cfg)

        wandb.config.update(merged_cfgs)


class SandBoxExecutor:
    """
    This SandBoxExecutor class is used to provide a sandbox environment for
        exploring data-model co-designs in a one-stop manner with fast feedback
         and tiny model size, small data size, and high efficiency.

        It plays as a middleware maintains the data-juicer's data executor,
        a model processor (training and inference), and an auto-evaluator,
        where the latter two ones are usually from third-party libraries.

    """

    def __init__(
        self,
        cfg=None,
    ):
        """
        Initialization method.

        :param cfg: configuration of sandbox.

        """
        self.cfg = cfg

        self.watcher = SandBoxWatcher(self.cfg)
        self.watcher.watch_cfgs([(cfg, 'sandbox')])

        # jobs to probe, refine_recipe, execution and evaluation for
        # interested data and model within the sandbox
        self.probe_jobs = []
        self.refine_recipe_jobs = []
        self.execution_jobs = []
        self.evaluation_jobs = []

        self.register_jobs()

    def register_jobs(self):

        # register probe_jobs
        for job_cfg in self.cfg.probe_job_configs:
            self.probe_jobs.append(register_hook(job_cfg, self.watcher))

        # register refine_recipe_jobs
        for job_cfg in self.cfg.refine_recipe_job_configs:
            self.refine_recipe_jobs.append(register_hook(
                job_cfg, self.watcher))

        # register execution_jobs
        for job_cfg in self.cfg.execution_job_configs:
            self.execution_jobs.append(register_hook(job_cfg, self.watcher))

        # register evaluation_jobs
        for job_cfg in self.cfg.evaluation_job_configs:
            self.evaluation_jobs.append(register_hook(job_cfg, self.watcher))

    def run(self):
        """
         Running the sandbox pipeline at once or in HPO style.
        """
        if self.cfg.hpo_config is not None:
            # execute_hpo_wandb contains running one_trail with HPO scheduler
            self.execute_hpo_wandb()
        else:
            self.one_trial()

    def one_trial(self):
        """
        Running the sandbox pipeline at once.
         Users can flexibly conduct some steps of the whole sandbox pipeline
          according to their own need and configuration. The watcher will
          automatically track the results in terms of data, model and specified
          evaluation metrics to the watcher.

        """
        # TODO: how the hpo work
        if self.watcher.object_name_in_hpo is not None:
            # merge the new hyper-parameters produced by HPO scheduler
            self.cfg = merge_config(self.cfg, wandb.config)
            self.watcher.watch_cfgs([self.cfg, 'after_hpo'])

        job_infos = {}

        # ====== Data & model probe ======
        for probe_hook in self.probe_jobs:
            job_infos = probe_hook.hook(**job_infos)

        # ====== Data-model recipes iteration based on probe results ======
        for refine_hook in self.refine_recipe_jobs:
            job_infos = refine_hook.hook(**job_infos)

        # ====== Data processing & model training ======
        for exec_hook in self.execution_jobs:
            job_infos = exec_hook.hook(**job_infos)

        # ====== Evaluation on processed data or trained model ======
        for eval_hook in self.evaluation_jobs:
            job_infos = eval_hook.hook(**job_infos)

    def execute_hpo_wandb(self):
        """
        Running the sandbox pipeline in HPO style.
         Users can flexibly conduct some steps of the whole sandbox pipeline
          according to their own need and configuration. The watcher will
          automatically track the results in terms of data, model and specified
          evaluation metrics to the watcher.
        """
        with open(self.cfg.hpo_config) as file:
            hpo_configuration = yaml.safe_load(file)
            sweep_id = self.watcher.setup_sweep(hpo_configuration)
            wandb.agent(sweep_id,
                        function=self.one_trial,
                        count=hpo_configuration['sweep_max_count']
                        if 'sweep_max_count' in hpo_configuration else None)
