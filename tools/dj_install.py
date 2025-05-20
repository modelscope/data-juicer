import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from data_juicer.utils.config_utils import init_configs
from data_juicer.utils.lazy_loader import LazyLoader

# Map operators to their required packages
OPS_TO_PKG = {
    # Add your operator to package mappings here
    # Example:
    # 'text_length_filter': ['numpy'],
    # 'image_quality_filter': ['opencv-python', 'numpy'],
}

# Paths to requirements files for version constraints
require_version_paths = [
    Path(__file__).parent.parent / 'requirements.txt',
    Path(__file__).parent.parent / 'requirements-dev.txt',
]


def main():
    parser = argparse.ArgumentParser(
        description='Install Data-Juicer dependencies')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to the config file to determine required operators')
    parser.add_argument(
        '--ops',
        type=str,
        nargs='+',
        help='Operator categories to install '
        '(e.g., vision, nlp, audio, generic, distributed, sandbox)')
    args = parser.parse_args()

    # If no specific ops or config provided, install all dependencies
    if not args.ops and not args.config:
        logger.info('Installing all dependencies...')
        try:
            LazyLoader.check_packages(['.'])
            logger.info('All dependencies installed successfully.')
        except Exception as e:
            logger.error(f'Failed to install dependencies: {e}')
            sys.exit(1)
        return

    # If config is provided, install based on operators in config
    if args.config:
        cfg = init_configs(args.config)

        # get the ops in the recipe
        op_names = [list(op.keys())[0] for op in cfg.process]
        recipe_reqs = []
        for op_name in op_names:
            if op_name in OPS_TO_PKG:
                recipe_reqs.extend(OPS_TO_PKG[op_name])
        recipe_reqs = list(set(recipe_reqs))

        # get the package version limit of Data-Juicer
        version_map, reqs = {}, []
        for path in require_version_paths:
            if not os.path.exists(path):
                logger.warning(f'target file does not exist: {path}')
            else:
                with open(path, 'r', encoding='utf-8') as fin:
                    reqs += [x.strip() for x in fin.read().splitlines()]
        for req in reqs:
            clean_req = req.replace('<', ' ').replace('>', ' ').replace(
                '=', ' ').split(' ')[0]
            version_map[clean_req] = req

        # generate require file for the recipe
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file_path = temp_file.name
            for req in recipe_reqs:
                if req in version_map:
                    temp_file.write(version_map[req] + '\n')
                else:
                    temp_file.write(req + '\n')

        # install by calling 'pip install -r ...'
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-r', temp_file_path])
            logger.info('Requirements were installed successfully.')
        except subprocess.CalledProcessError as e:
            logger.error(
                f'An error occurred while installing the requirements: {e}')
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            sys.exit(1)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        return

    # Install specific operator categories
    for op_category in args.ops:
        logger.info(f'Installing dependencies for {op_category}...')
        try:
            LazyLoader.check_packages([f'.[{op_category}]'])
            logger.info(
                f'Dependencies for {op_category} installed successfully.')
        except Exception as e:
            logger.error(
                f'Failed to install dependencies for {op_category}: {e}')
            sys.exit(1)


if __name__ == '__main__':
    main()
