import os
import subprocess
import sys
import tempfile

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.utils.auto_install_mapping import OPS_TO_PKG

require_version_paths = ['./environments/science_requires.txt']


def main():
    cfg = init_configs()

    # get the ops in the recipe
    op_names = [list(op.keys())[0] for op in cfg.process]
    recipe_reqs = []
    for op_name in op_names:
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
        clean_req = req.replace('<',
                                ' ').replace('>',
                                             ' ').replace('=',
                                                          ' ').split(' ')[0]
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
        logger.info(
            f'An error occurred while installing the requirements: {e}')
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        sys.exit(1)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == '__main__':
    main()
