import copy
import sys

from loguru import logger

from data_juicer.config import export_config, init_configs
from data_juicer.core import Analyser, Executor
from data_juicer.utils.constant import StatsKeys


@logger.catch
def main():

    path_3sigma_recipe = None
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == '--path_3sigma_recipe':
            path_3sigma_recipe = sys.argv[i + 1]

    # 1. analyze using the given initial recipe
    cfg = init_configs()
    logger.info('Begin to analyze data using the given initial recipe')

    analyser = Analyser(cfg)
    analyser.run()
    df = analyser.overall_result
    # get the mapping from op_name to their mu and sigma
    mean_series = df[df.index == 'mean']
    stats_key_to_mean = mean_series.iloc[0, :].to_dict()
    std_series = df[df.index == 'std']
    stats_key_to_std = std_series.iloc[0, :].to_dict()

    # 2. adjust the hyper-parameters of the given recipe with 3-sigma rule
    logger.info('Begin to modify the recipe with 3-sigma rule')
    op_name_to_stats_key = StatsKeys.get_access_log(dj_cfg=cfg)
    for process in cfg.process:
        op_name, args = list(process.items())[0]
        temp_args = copy.deepcopy(args)
        if op_name not in op_name_to_stats_key:
            # skip the op such as `clean_email_mapper`
            continue
        stats_keys = op_name_to_stats_key[op_name]
        for stats_key in stats_keys:
            if stats_key in stats_key_to_mean:
                for arg_name in temp_args.keys():
                    new_val = None
                    if 'min' in arg_name:
                        new_val = stats_key_to_mean[stats_key] - \
                                  3 * stats_key_to_std[stats_key]
                    if 'max' in arg_name:
                        new_val = stats_key_to_mean[stats_key] + \
                                  3 * stats_key_to_std[stats_key]
                    if new_val is not None and str(new_val) != 'nan':
                        logger.info(f'Using 3-sigma rule, for op {op_name}, '
                                    f'changed its para '
                                    f'{arg_name}={args[arg_name]} into '
                                    f'{arg_name}={new_val}')
                        args[arg_name] = new_val

    if path_3sigma_recipe:
        export_config(cfg, path_3sigma_recipe)

    # 3. process the data using the refined recipe
    logger.info('Begin to process the data with refined recipe')
    if cfg.executor_type == 'default':
        executor = Executor(cfg)
    elif cfg.executor_type == 'ray':
        from data_juicer.core.ray_executor import RayExecutor
        executor = RayExecutor(cfg)
    executor.run()


if __name__ == '__main__':
    main()
