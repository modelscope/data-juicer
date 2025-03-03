from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    if cfg.executor_type == 'default':
        executor = DefaultExecutor(cfg)
    elif cfg.executor_type == 'ray':
        from data_juicer.core.executor.ray_executor import RayExecutor
        executor = RayExecutor(cfg)
    executor.run()


if __name__ == '__main__':
    main()
