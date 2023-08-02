from loguru import logger

from data_juicer.core import Executor


@logger.catch
def main():
    executor = Executor()
    executor.run()


if __name__ == '__main__':
    main()
