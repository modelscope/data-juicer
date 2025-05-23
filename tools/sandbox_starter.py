from argparse import ArgumentError
from typing import List, Union

from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from data_juicer.core.sandbox.pipelines import SandBoxExecutor


def init_sandbox_configs(args=None):
    """
    initialize the jsonargparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. config files in yaml (json and jsonnet supersets);
        3. environment variables
        4. hard-coded defaults

    :param args: list of params, e.g., ['--conifg', 'cfg.yaml'], default None.
    :return: a global cfg object used by the Executor or Analyzer
    """
    parser = ArgumentParser(default_env=True, default_config_files=None)

    parser.add_argument('--config',
                        action=ActionConfigFile,
                        help='Path to a dj basic configuration file.',
                        required=True)

    parser.add_argument('--project_name',
                        type=str,
                        default='hello_world',
                        help='Name of your data process project.')

    parser.add_argument('--experiment_name',
                        type=str,
                        default='experiment1',
                        help='For wandb tracer name.')

    parser.add_argument('--work_dir',
                        type=str,
                        default='./outputs/hello_world',
                        help='Default output dir of meta information.')

    parser.add_argument(
        '--hpo_config',
        type=str,
        help='Path to a configuration file when using auto-HPO tool.',
        required=False)

    parser.add_argument('--pipelines',
                        type=List[dict],
                        default=[],
                        help='List of pipelines.')

    parser.add_argument('--probe_job_configs',
                        type=Union[List[str], List[dict]],
                        default=[],
                        help='List of params for each probe job.')

    parser.add_argument('--refine_recipe_job_configs',
                        type=Union[List[str], List[dict]],
                        default=[],
                        help='List of params for each refine-recipe jobs.')

    parser.add_argument('--execution_job_configs',
                        type=Union[List[str], List[dict]],
                        default=[],
                        help='List of params for each execution jobs.')

    parser.add_argument('--evaluation_job_configs',
                        type=Union[List[str], List[dict]],
                        default=[],
                        help='List of params for each evaluation jobs.')

    try:
        cfg = parser.parse_args(args=args)

        if cfg.pipelines and (cfg.probe_job_configs
                              or cfg.refine_recipe_job_configs
                              or cfg.execution_job_configs
                              or cfg.evaluation_job_configs):
            logger.error(
                'Cannot specify both pipelines and top-level job configs')
            exit(1)

        return cfg
    except ArgumentError:
        logger.error('Config initialization failed')


@logger.catch
def main():
    cfg = init_sandbox_configs()
    sandbox_executor = SandBoxExecutor(cfg)
    sandbox_executor.run()


if __name__ == '__main__':
    main()
