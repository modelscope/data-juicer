from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import Executor, RayExecutor


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    if cfg.executor_type == 'default':
        executor = Executor(cfg)
    elif cfg.executor_type == 'ray':
        executor = RayExecutor(cfg)
    executor.run()


if __name__ == '__main__':
    main()
