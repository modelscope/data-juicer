import argparse
import sys

from loguru import logger

from data_juicer.utils.lazy_loader import LazyLoader


def main():
    parser = argparse.ArgumentParser(description='Install Data-Juicer dependencies')
    parser.add_argument('--ops', type=str, nargs='+', help='Operator categories to install (e.g., vision, nlp, audio, generic, distributed, sandbox)')
    args = parser.parse_args()

    # If no specific ops provided, install all dependencies
    if not args.ops:
        logger.info('Installing all dependencies...')
        try:
            LazyLoader.check_packages(['.'])
            logger.info('All dependencies installed successfully.')
        except Exception as e:
            logger.error(f'Failed to install dependencies: {e}')
            sys.exit(1)
        return

    # Install specific operator categories
    for op_category in args.ops:
        logger.info(f'Installing dependencies for {op_category}...')
        try:
            LazyLoader.check_packages([f'.[{op_category}]'])
            logger.info(f'Dependencies for {op_category} installed successfully.')
        except Exception as e:
            logger.error(f'Failed to install dependencies for {op_category}: {e}')
            sys.exit(1)


if __name__ == '__main__':
    main()
