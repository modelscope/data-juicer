import os
from typing import List


def validate_hook_output(pipelines, output_key):
    """
    Validate whether a specified hook output is valid

    This function parses the output_key and searches for the corresponding hook
    within the given pipeline list, then validates whether the hook contains
    the specified output key.

    :param pipelines: A list of pipeline objects, each containing a name attribute
        and job lists including probe_jobs, refine_recipe_jobs, execution_jobs, and evaluation_jobs
    :param output_key: The output key string with format "pipeline_name.hook_meta_name.output_name"
    :return: True if the corresponding pipeline, hook and output key are found and valid, otherwise False
    """
    pipeline_name, hook_meta_name, output_name = output_key.split(".")
    for pipeline in pipelines:
        if pipeline.name == pipeline_name:
            all_jobs = (
                pipeline.probe_jobs + pipeline.refine_recipe_jobs + pipeline.execution_jobs + pipeline.evaluation_jobs
            )
            for hook in all_jobs:
                if hook.meta_name == hook_meta_name:
                    if hook.output_keys is None or output_name in hook.output_keys:
                        return True
    return False


def guess_file_or_dir(path: str) -> str:
    """
    Guess a path is a file or a directory.

    If there is a "." in the basename of the path and the "." is not the first char, guess it's a file.
    Otherwise, guess it's a directory.
    """
    clean_path = path.rstrip("/\\")
    basename = os.path.basename(clean_path)

    if "." in basename and not basename.startswith("."):
        return "file"
    else:
        return "dir"


def add_iter_subdir_to_paths(paths: List[str], iter_num: int) -> List[str]:
    """
    Add iteration number as a subdir to the specified paths.

    Example:
        1. files: "/a/b/c/d.jsonl" --> "/a/b/c/{iter_num}/d.jsonl"
        2. dirs: "/a/b/c" --> "/a/b/c/{iter_num}"

    :param paths: the input original paths
    :param iter_num: the iteration number to be added to the paths
    :return: the result paths with the same number as the original paths,
        with iteration numbers are added as the examples show.
    """
    res_paths = []
    for path in paths:
        if guess_file_or_dir(path) == "file":
            res_paths.append(os.path.join(os.path.dirname(path), f"iter_{str(iter_num)}", os.path.basename(path)))
        else:
            res_paths.append(os.path.join(path, f"iter_{str(iter_num)}"))

    return res_paths
