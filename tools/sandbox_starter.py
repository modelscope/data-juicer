import json

import yaml
from jsonargparse import dict_to_namespace
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core.sandbox.pipelines import SandBoxExecutor


def prepare_side_configs(config):
    if isinstance(config, str):
        # config path
        if config.endswith('.yaml') or config.endswith('.yml'):
            with open(config) as fin:
                config = yaml.safe_load(fin)
                return dict_to_namespace(config)
        elif config.endswith('.json'):
            with open(config) as fin:
                config = json.load(fin)
                return dict_to_namespace(config)
        else:
            raise TypeError(f'Unrecognized config file type [{config}]. '
                            f'Should be one of the types [".yaml", ".yml", '
                            f'".json"].')
    elif isinstance(config, dict):
        # config dict
        config = dict_to_namespace(config)
        return config
    else:
        raise TypeError(f'Unrecognized side config type: [{type(config)}.')


def split_configs(cfg):
    """
    Split train/infer/eval configs from the original config. Other configs can
    be specified by their dict objects or config file path strings.

    :param cfg: the original config
    :return: a dict of different configs.
    """
    configs = {
        'dj_cfg': cfg,
    }
    if cfg.model_infer_config:
        configs['model_infer_cfg'] = prepare_side_configs(
            cfg.model_infer_config)
    if cfg.model_train_config:
        configs['model_train_cfg'] = prepare_side_configs(
            cfg.model_train_config)
    if cfg.data_eval_config:
        configs['data_eval_cfg'] = prepare_side_configs(cfg.data_eval_config)
    if cfg.model_eval_config:
        configs['model_eval_cfg'] = prepare_side_configs(cfg.model_eval_config)

    return configs


@logger.catch
def main():
    cfg = init_configs()
    configs = split_configs(cfg)
    sandbox_executor = SandBoxExecutor(**configs)
    sandbox_executor.run()


if __name__ == '__main__':
    main()
