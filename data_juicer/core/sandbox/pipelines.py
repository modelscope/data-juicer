import json
import os
from copy import deepcopy
from typing import List

import wandb
import yaml
from jsonargparse import Namespace as JsonNamespace
from jsonargparse import dict_to_namespace, namespace_to_dict
from loguru import logger

from data_juicer.config import merge_config, prepare_side_configs
from data_juicer.core.sandbox.hooks import register_hook
from data_juicer.utils.constant import JobRequiredKeys


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
        if hpo_config is not None and "metric" in hpo_config and "name" in hpo_config["metric"]:
            self.object_name_in_hpo = hpo_config["metric"]["name"]
        else:
            self.object_name_in_hpo = None
        self.logged_res = {}

    def query(self, meta_name: str):
        """
        Query the result from the logged_res.
        """
        return self.logged_res.get(meta_name)

    def watch(self, res, meta_name: str = ""):
        """
        Flatten the result in dot structure and log it into WandB.
        """
        if isinstance(res, dict):
            for key, value in res.items():
                # getting the left nodes of the given res dictionary.
                if isinstance(value, dict):
                    self.watch(value, f"{meta_name}.{key}")
                else:
                    self.logged_res[f"{meta_name}.{key}"] = value
                    if self.object_name_in_hpo == f"{meta_name}.{key}":
                        # Ensuring float results for HPO experiments
                        value = float(value)
                    self.wandb_run.log({f"{meta_name}.{key}": value})
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
        if hpo_config is not None and "metric" in hpo_config and "name" in hpo_config["metric"]:
            self.object_name_in_hpo = hpo_config["metric"]["name"]
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
                    raise ValueError(f"Expected dict or JsonNamespace, got {type(cfg)}")
                for key, val in converged_cfg.items():
                    merged_cfgs[f"{cfg_prefix}.{key}"] = val
        else:
            merged_cfgs = namespace_to_dict(self.sandbox_cfg)

        wandb.config.update(merged_cfgs)


class SandboxPipeline:
    def __init__(self, pipeline_name="anonymous", pipeline_cfg=None, watcher=None):
        """
        Initialization method.
        """
        self.name = pipeline_name
        self.cfg = pipeline_cfg
        self.watcher = watcher

        # jobs to probe, refine_recipe, execution and evaluation for
        # interested data and model within the sandbox
        self.probe_jobs = []
        self.refine_recipe_jobs = []
        self.execution_jobs = []
        self.evaluation_jobs = []

        self.register_jobs()

    def register_jobs(self):
        # register probe_jobs
        for job_cfg in self.cfg.get("probe_job_configs", []):
            self.probe_jobs.append(register_hook(job_cfg, self.watcher))

        # register refine_recipe_jobs
        for job_cfg in self.cfg.get("refine_recipe_job_configs", []):
            self.refine_recipe_jobs.append(register_hook(job_cfg, self.watcher))

        # register execution_jobs
        for job_cfg in self.cfg.get("execution_job_configs", []):
            self.execution_jobs.append(register_hook(job_cfg, self.watcher))

        # register evaluation_jobs
        for job_cfg in self.cfg.get("evaluation_job_configs", []):
            self.evaluation_jobs.append(register_hook(job_cfg, self.watcher))

    def run(self, **pipeline_infos):
        """
        Running the sandbox pipeline at once or in HPO style.
        """
        if self.cfg.hpo_config is not None:
            # execute_hpo_wandb contains running one_trail with HPO scheduler
            return self.execute_hpo_wandb(**pipeline_infos)
        else:
            return self.one_trial(**pipeline_infos)

    def one_trial(self, **context_infos):
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
            self.watcher.watch_cfgs([self.cfg, "after_hpo"])

        if self.name in context_infos:
            raise ValueError(f"There are different pipelines with the same pipeline name {self.name}.")
        context_infos[self.name] = []

        # ====== Data & model probe ======
        for probe_hook in self.probe_jobs:
            logger.info(f"======= Pipeline [{self.name}]: Start Probe Hook [{probe_hook.meta_name}] =======")
            new_job_infos = probe_hook.run(**context_infos)
            context_infos[self.name].append(new_job_infos)

        # ====== Data-model recipes iteration based on probe results ======
        for refine_hook in self.refine_recipe_jobs:
            logger.info(f"======= Pipeline [{self.name}]: Start Refine Hook [{refine_hook.meta_name}] =======")
            new_job_infos = refine_hook.run(**context_infos)
            context_infos[self.name].append(new_job_infos)

        # ====== Data processing & model training ======
        for exec_hook in self.execution_jobs:
            logger.info(f"======= Pipeline [{self.name}]: Start Execution Hook [{exec_hook.meta_name}] =======")
            new_job_infos = exec_hook.run(**context_infos)
            context_infos[self.name].append(new_job_infos)

        # ====== Evaluation on processed data or trained model ======
        for eval_hook in self.evaluation_jobs:
            logger.info(f"======= Pipeline [{self.name}]: Start Evaluation Hook [{eval_hook.meta_name}] =======")
            new_job_infos = eval_hook.run(**context_infos)
            context_infos[self.name].append(new_job_infos)

        return context_infos

    def execute_hpo_wandb(self, **pipeline_infos):
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
            wandb.agent(
                sweep_id,
                function=self.one_trial,
                count=hpo_configuration["sweep_max_count"] if "sweep_max_count" in hpo_configuration else None,
            )
        return None


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
        self.watcher.watch_cfgs([(cfg, "sandbox")])

        self.pipelines = self.parse_pipelines(self.cfg)

        self.resume = self.cfg.get("resume", False)

    def parse_pipelines(self, cfg):
        """
        Parse the pipeline configs.

        :param cfg: the original config
        :return: a list of SandBoxPipeline objects.
        """
        pipelines = []
        pipeline_keys = [
            "pipelines",
            "probe_job_configs",
            "refine_recipe_job_configs",
            "execution_job_configs",
            "evaluation_job_configs",
        ]
        global_cfgs = deepcopy(cfg)
        for pipeline_key in pipeline_keys:
            if pipeline_key in global_cfgs:
                global_cfgs.pop(pipeline_key)
        if cfg.pipelines:
            # specify the pipelines
            for pipeline in cfg.pipelines:
                pipeline_name, pipeline_cfg = list(pipeline.items())[0]
                pipeline_cfg.update(global_cfgs)
                pipelines.append(SandboxPipeline(pipeline_name, self.specify_jobs_configs(pipeline_cfg), self.watcher))
        else:
            pipeline = SandboxPipeline(pipeline_cfg=self.specify_jobs_configs(cfg), watcher=self.watcher)
            pipelines.append(pipeline)
        return pipelines

    def specify_job_configs(self, ori_config):
        config = prepare_side_configs(ori_config)

        for key in JobRequiredKeys:
            if key.value not in config:
                logger.debug(f'The key "{key.value}" is not specified in {ori_config}')

        return dict_to_namespace(config)

    def specify_jobs_configs(self, cfg):
        """
        Specify job configs by their dict objects or config file path strings.

        :param cfg: the original config
        :return: a dict of different configs.
        """

        def configs_to_job_list(cfgs):
            job_cfgs = []
            if cfgs:
                job_cfgs = [self.specify_job_configs(job_cfg) for job_cfg in cfgs]
            return job_cfgs

        if isinstance(cfg, dict):
            cfg = dict_to_namespace(cfg)

        if "probe_job_configs" in cfg:
            cfg.probe_job_configs = configs_to_job_list(cfg.probe_job_configs)
        if "refine_recipe_job_configs" in cfg:
            cfg.refine_recipe_job_configs = configs_to_job_list(cfg.refine_recipe_job_configs)
        if "execution_job_configs" in cfg:
            cfg.execution_job_configs = configs_to_job_list(cfg.execution_job_configs)
        if "evaluation_job_configs" in cfg:
            cfg.evaluation_job_configs = configs_to_job_list(cfg.evaluation_job_configs)

        return cfg

    def run(self):
        context_infos_path = os.path.join(self.cfg.work_dir, "context_infos.json")
        if self.resume and os.path.exists(context_infos_path):
            # load context infos from the existing one
            context_infos = json.load(open(context_infos_path, "r"))
            # find those finished pipelines
            finished_pipelines = set(context_infos.keys())
            left_pipelines = []
            for pipeline in self.pipelines:
                # check if the pipeline is already existing in the context infos
                if pipeline.name in finished_pipelines:
                    # check if the number of job infos is the same as the number of all kinds of jobs,
                    # which means all jobs are finished
                    num_job_infos = len(context_infos[pipeline.name])
                    num_jobs = (
                        len(pipeline.probe_jobs)
                        + len(pipeline.refine_recipe_jobs)
                        + len(pipeline.execution_jobs)
                        + len(pipeline.evaluation_jobs)
                    )
                    if num_job_infos == num_jobs:
                        logger.info(
                            f"Pipeline {pipeline.name} is finished and loaded from the existing context infos. Skip it!"
                        )
                        continue
                left_pipelines.append(pipeline)
            self.pipelines = left_pipelines
        else:
            context_infos = {}

        try:
            for pipeline in self.pipelines:
                context_infos = pipeline.run(**context_infos)
        finally:
            # export context infos
            with open(context_infos_path, "w") as fout:
                json.dump(context_infos, fout, indent=4)
