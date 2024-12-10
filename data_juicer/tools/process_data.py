from loguru import logger
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
from data_juicer.config import init_configs
from data_juicer.core import Executor

@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    if cfg.executor_type == 'default':
        executor = Executor(cfg)
    elif cfg.executor_type == 'ray':
        from data_juicer.core.ray_executor import RayExecutor
        executor = RayExecutor(cfg)
    executor.run()


if __name__ == '__main__':
    main()
