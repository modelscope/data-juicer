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
from data_juicer.core.sandbox.context_infos import (
    ContextInfos,
    GlobalContextInfos,
    PipelineInfos,
)
from data_juicer.core.sandbox.hooks import register_hook
from data_juicer.core.sandbox.utils import validate_hook_output
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


class Target:

    SUPPORT_OPS = ["==", ">=", "<=", ">", "<"]

    key: str
    op: str
    tgt_val: float

    def __init__(self, iter_target_str: str = None, key: str = None, op: str = None, tgt_val: float = None):
        if iter_target_str is not None:
            self.parse_iter_targets(iter_target_str)
        else:
            self.key = key
            self.op = op
            self.tgt_val = tgt_val

    def parse_iter_targets(self, iter_target_str):
        for op in self.SUPPORT_OPS:
            if op in iter_target_str:
                target_key, target_value = [s.strip() for s in iter_target_str.split(op)]
                # check if the target value is a number
                try:
                    target_value = float(target_value)
                except:  # noqa: E722
                    logger.error(
                        f"Invalid iter_targets [{iter_target_str}]: The target value [{target_value}] "
                        "is not a valid number."
                    )
                    exit(1)
                self.key = target_key
                self.op = op
                self.tgt_val = target_value
                break
        else:
            logger.error(
                f"Invalid iter_targets [{iter_target_str}]: No valid comparators are found."
                f"Only support {self.SUPPORT_OPS}"
            )
            exit(1)

    def check_target(self, context_infos: ContextInfos):
        curr_val = context_infos[self.key]
        try:
            logger.debug(f"Checking {curr_val} {self.op} {self.tgt_val}")
            ret = eval(f"{curr_val} {self.op} {self.tgt_val}")
        except:  # noqa: E722
            logger.error(f"Invalid iter_targets [{str(self)}]: The target value [{curr_val}] " "is not a valid number.")
            return False
        return ret

    def __str__(self):
        return f"{self.key} {self.op} {self.tgt_val}"


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

    def run(self, context_infos: ContextInfos):
        """
        Running the sandbox pipeline at once or in HPO style.
        """
        if self.cfg.hpo_config is not None:
            # execute_hpo_wandb contains running one_trail with HPO scheduler
            return self.execute_hpo_wandb(context_infos)
        else:
            return self.one_trial(context_infos)

    def one_trial(self, context_infos: ContextInfos):
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

        if self.name in context_infos.pipeline_names:
            raise ValueError(f"There are different pipelines with the same pipeline name {self.name}.")
        pipeline_infos = PipelineInfos(self.name)
        context_infos.record_pipeline_infos(pipeline_infos)

        # ====== Data & model probe ======
        for probe_hook in self.probe_jobs:
            logger.info(
                f"======= Iter [{context_infos.iter}] - Pipeline [{self.name}]: Start Probe Hook [{probe_hook.meta_name}] ======="
            )
            new_job_infos = probe_hook.run(context_infos)
            context_infos[self.name].record_job_infos(new_job_infos)
            logger.debug(f"Context Infos: {context_infos.to_dict()}")

        # ====== Data-model recipes iteration based on probe results ======
        for refine_hook in self.refine_recipe_jobs:
            logger.info(
                f"======= Iter [{context_infos.iter}] - Pipeline [{self.name}]: Start Refine Hook [{refine_hook.meta_name}] ======="
            )
            new_job_infos = refine_hook.run(context_infos)
            context_infos[self.name].record_job_infos(new_job_infos)
            logger.debug(f"Context Infos: {context_infos.to_dict()}")

        # ====== Data processing & model training ======
        for exec_hook in self.execution_jobs:
            logger.info(
                f"======= Iter [{context_infos.iter}] - Pipeline [{self.name}]: Start Execution Hook [{exec_hook.meta_name}] ======="
            )
            new_job_infos = exec_hook.run(context_infos)
            context_infos[self.name].record_job_infos(new_job_infos)
            logger.debug(f"Context Infos: {context_infos.to_dict()}")

        # ====== Evaluation on processed data or trained model ======
        for eval_hook in self.evaluation_jobs:
            logger.info(
                f"======= Iter [{context_infos.iter}] - Pipeline [{self.name}]: Start Evaluation Hook [{eval_hook.meta_name}] ======="
            )
            new_job_infos = eval_hook.run(context_infos)
            context_infos[self.name].record_job_infos(new_job_infos)
            logger.debug(f"Context Infos: {context_infos.to_dict()}")

        return context_infos

    def execute_hpo_wandb(self, context_infos):
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

        # iterative related
        self.max_iter_num = self.cfg.get("max_iter_num", 1)
        init_targets = self.cfg.get("iter_targets", [])
        # if both of them are not set
        if self.max_iter_num < 0:
            logger.error(f"Argument 'max_iter_num' must be 0 or a positive number. Got [{self.max_iter_num}].")
            exit(1)
        if not isinstance(init_targets, list):
            init_targets = [init_targets]
        if self.max_iter_num == 0 and len(init_targets) == 0:
            logger.error(
                "Either 'max_iter_num' must be > 0 or 'iter_targets' must be set. "
                "If you want to run the pipeline without iterative, please leave both arguments at their default values"
                " or set 'max_iter_num' to 1."
            )
            exit(1)

        init_targets = [Target(iter_target_str=iter_target_str) for iter_target_str in init_targets]

        self.iter_targets = []
        for target in init_targets:
            if not validate_hook_output(self.pipelines, target.key):
                logger.error(
                    f"Invalid iter_targets [{str(target)}]: "
                    f"The target metric key [{target.key}] can not found in the pipelines."
                )
            self.iter_targets.append(target)

        self.iter_targets_mode = self.cfg.get("iter_targets_mode", "all")

        # iterative updater for config arguments
        self.iter_updater = self.cfg.get("iter_updater", {})

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

    def iterative_update_pipelines(self, current_pipelines: List[SandboxPipeline], last_context_infos: ContextInfos):
        if current_pipelines is None:
            return None
        if last_context_infos is None or len(last_context_infos) == 0:
            return current_pipelines

        # get the pipeline configs
        for from_key, target_key in self.iter_updater.items():
            from_value = last_context_infos[from_key]
            if from_value is not None:
                cfg_levels = target_key.split(".")
                if len(cfg_levels) < 4:
                    raise ValueError(
                        f"The target key [{target_key}] must be in the format of "
                        f"<pipeline_name>.<hook_meta_name>.[extra_configs|dj_configs].<hook_cfg_key1>[.<hook_cfg_keyn>]."
                    )
                tgt_pipeline_name = cfg_levels[0]
                tgt_hook_meta_name = cfg_levels[1]
                tgt_local_key = ".".join(cfg_levels[2:])
                for i in range(len(current_pipelines)):
                    current_pipeline = current_pipelines[i]
                    if current_pipeline.name == tgt_pipeline_name:
                        all_hooks = (
                            current_pipeline.probe_jobs
                            + current_pipeline.refine_recipe_jobs
                            + current_pipeline.execution_jobs
                            + current_pipeline.evaluation_jobs
                        )
                        for hook in all_hooks:
                            if hook.meta_name == tgt_hook_meta_name:
                                # put the updated configs key/values into the local settings
                                hook.local_settings[tgt_local_key] = from_value
                    current_pipelines[i] = current_pipeline
            else:
                logger.warning(f"The iter_updater [{from_key}] is not found in the last context infos.")
        return current_pipelines

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
        num_pipeline_skip = 0
        last_context_infos = ContextInfos(iter=0)
        if self.resume and os.path.exists(context_infos_path):
            # load context infos from the existing one
            context_infos_list = json.load(open(context_infos_path, "r"))
            context_infos_list = GlobalContextInfos.from_list(context_infos_list)
            current_iter = len(context_infos_list)
            if current_iter == 0:
                logger.info("The context infos file is empty. Start from the first iter.")
            else:
                logger.info(f"Continue from the iter {current_iter}.")
                current_iter -= 1
                last_context_infos = context_infos_list[-1]
                context_infos_list = context_infos_list[:-1]
                # find those finished pipelines
                finished_pipelines = set(last_context_infos.pipeline_names)
                for pipeline in self.pipelines:
                    # check if the pipeline is already existing in the context infos
                    if pipeline.name in finished_pipelines:
                        # check if the number of job infos is the same as the number of all kinds of jobs,
                        # which means all jobs are finished
                        num_job_infos = len(last_context_infos[pipeline.name])
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
                            num_pipeline_skip += 1
                            continue
        else:
            context_infos_list = GlobalContextInfos()
            current_iter = 0

        try:
            current_pipelines = deepcopy(self.pipelines)
            while True:
                current_iter += 1
                logger.info(f"============== Starting the iter {current_iter} ==============")
                if num_pipeline_skip > 0:
                    context_infos = last_context_infos
                else:
                    context_infos = ContextInfos(iter=current_iter)
                for pipeline in current_pipelines:
                    if num_pipeline_skip > 0:
                        num_pipeline_skip -= 1
                        continue
                    context_infos = pipeline.run(context_infos)
                context_infos_list.record_context_infos(context_infos)

                # check if the pipelines reach the max number of iterations
                if 0 < self.max_iter_num <= current_iter:
                    break
                # check if the running meet the targets
                if len(self.iter_targets) > 0:
                    curr_target_results = [iter_target.check_target(context_infos) for iter_target in self.iter_targets]
                    if self.iter_targets_mode == "all":
                        if all(curr_target_results):
                            logger.info("All targets are satisfied.")
                            break
                    elif self.iter_targets_mode == "any":
                        if any(curr_target_results):
                            satisfied_idxes = [
                                idx for idx, curr_target_result in enumerate(curr_target_results) if curr_target_result
                            ]
                            satisfied_targets = [str(self.iter_targets[idx]) for idx in satisfied_idxes]
                            logger.info(f"Targets {satisfied_targets} are satisfied.")
                            break

                # check if there are any arguments to be updated from the last iteration
                if len(self.iter_updater) > 0:
                    logger.info("Updating arguments across iterations...")
                    current_pipelines = deepcopy(self.pipelines)
                    current_pipelines = self.iterative_update_pipelines(current_pipelines, context_infos)
        finally:
            # export context infos
            with open(context_infos_path, "w") as fout:
                json.dump(context_infos_list.to_list(), fout, indent=4)
