from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import SandBoxExecutor


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
        configs['model_infer_cfg'] = cfg.model_infer_config
    if cfg.model_train_config:
        configs['model_train_cfg'] = cfg.model_train_config
    if cfg.data_eval_config:
        configs['data_eval_cfg'] = cfg.data_eval_config
    if cfg.model_eval_config:
        configs['model_eval_cfg'] = cfg.model_eval_config

    return configs


@logger.catch
def main():
    cfg = init_configs()
    configs = split_configs(cfg)
    sandbox_executor = SandBoxExecutor(**configs)
    sandbox_executor.run()


if __name__ == '__main__':
    main()
