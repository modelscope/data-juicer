def validate_hook_output(pipelines, output_key):
    pipeline_name, hook_meta_name, output_name = output_key.split(".")
    for pipeline in pipelines:
        if pipeline.name == pipeline_name:
            all_jobs = (
                pipeline.probe_jobs + pipeline.refine_recipe_jobs + pipeline.execution_jobs + pipeline.evaluation_jobs
            )
            for hook in all_jobs:
                if hook.meta_name == hook_meta_name:
                    if hook.outputs is None or output_name in hook.output_keys:
                        return True
    return False
