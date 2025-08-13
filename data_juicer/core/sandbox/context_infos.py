from typing import Any, List


class JobInfos:
    meta_name: str
    output_keys: List[str]
    outputs: List[Any]

    def __init__(self, meta_name: str, output_keys: List[str], outputs: List[Any]):
        assert len(output_keys) == len(outputs)
        self.meta_name = meta_name
        self.output_keys = output_keys
        self.outputs = outputs

    def __getitem__(self, item):
        """
        "item" should be a string of output_key
        """
        if item not in self.output_keys:
            return None
        output_idx = self.output_keys.index(item)
        return self.outputs[output_idx]

    def __len__(self):
        return len(self.output_keys)

    def to_dict(self):
        ret = {"meta_name": self.meta_name}
        for output_key, output in zip(self.output_keys, self.outputs):
            ret[output_key] = output
        return ret

    @classmethod
    def from_dict(cls, job_infos_dict):
        job_infos_dict = job_infos_dict.copy()
        meta_name = job_infos_dict.pop("meta_name")
        output_keys = list(job_infos_dict.keys())
        outputs = [job_infos_dict[output_key] for output_key in output_keys]
        return cls(meta_name, output_keys, outputs)


class PipelineInfos:
    pipeline_name: str
    job_meta_names: List[str]
    job_infos: List[JobInfos]

    def __init__(self, pipeline_name: str, job_meta_names: List[str] = None, job_infos: List[JobInfos] = None):
        if job_meta_names is None:
            job_meta_names = []
        if job_infos is None:
            job_infos = []
        assert len(job_meta_names) == len(job_infos)
        self.pipeline_name = pipeline_name
        self.job_meta_names = job_meta_names
        self.job_infos = job_infos

    def record_job_infos(self, job_infos: JobInfos):
        self.job_meta_names.append(job_infos.meta_name)
        self.job_infos.append(job_infos)

    def get_the_last_job_infos(self):
        return self.job_infos[-1]

    def __getitem__(self, item):
        parts = item.split(".")
        job_meta_name, left_keys = parts[0], ".".join(parts[1:])
        if job_meta_name not in self.job_meta_names:
            return None
        job_idx = self.job_meta_names.index(job_meta_name)
        ret = self.job_infos[job_idx]
        if left_keys != "":
            ret = ret[left_keys]
        return ret

    def __len__(self):
        return len(self.job_meta_names)

    def to_dict(self):
        ret = {
            "pipeline_name": self.pipeline_name,
            "job_infos": [job_info.to_dict() for job_info in self.job_infos],
        }
        return ret

    @classmethod
    def from_dict(cls, pipeline_context_infos_dict):
        pipeline_name = pipeline_context_infos_dict.pop("pipeline_name")
        job_infos = [JobInfos.from_dict(job_info) for job_info in pipeline_context_infos_dict["job_infos"]]
        job_meta_names = [job_info.meta_name for job_info in job_infos]
        return cls(pipeline_name, job_meta_names, job_infos)


class ContextInfos:
    iter: int
    pipeline_names: List[str]
    pipeline_infos: List[PipelineInfos]

    def __init__(self, iter: int, pipeline_names=None, pipeline_infos=None):
        if pipeline_infos is None:
            pipeline_infos = []
        if pipeline_names is None:
            pipeline_names = []
        assert len(pipeline_names) == len(pipeline_infos)
        self.iter = iter
        self.pipeline_names = pipeline_names
        self.pipeline_infos = pipeline_infos

    def record_pipeline_infos(self, pipeline_infos: PipelineInfos):
        self.pipeline_names.append(pipeline_infos.pipeline_name)
        self.pipeline_infos.append(pipeline_infos)

    def get_the_last_job_infos(self):
        last_idx = len(self.pipeline_infos) - 1
        while len(self.pipeline_infos[last_idx]) == 0:
            last_idx -= 1
            if last_idx < 0:
                raise ValueError("Cannot find the last non-empty job infos.")
        return self.pipeline_infos[last_idx].get_the_last_job_infos()

    def __getitem__(self, item):
        parts = item.split(".")
        pipeline_name, left_keys = parts[0], ".".join(parts[1:])
        if pipeline_name not in self.pipeline_names:
            return None
        pipeline_idx = self.pipeline_names.index(pipeline_name)
        ret = self.pipeline_infos[pipeline_idx]
        if left_keys != "":
            ret = ret[left_keys]
        return ret

    def __len__(self):
        return len(self.pipeline_names)

    def to_dict(self):
        ret = {
            "iter": self.iter,
            "pipeline_infos": [pipeline_infos.to_dict() for pipeline_infos in self.pipeline_infos],
        }
        return ret

    @classmethod
    def from_dict(cls, context_infos_dict):
        iter = context_infos_dict.pop("iter")
        pipeline_infos = [
            PipelineInfos.from_dict(pipeline_infos_dict) for pipeline_infos_dict in context_infos_dict["pipeline_infos"]
        ]
        pipeline_names = [pipeline_infos.pipeline_name for pipeline_infos in pipeline_infos]
        return cls(iter, pipeline_names, pipeline_infos)


class GlobalContextInfos:
    context_infos: List[ContextInfos]

    def __init__(self, context_infos=None):
        if context_infos is None:
            context_infos = []
        self.context_infos = context_infos

    def record_context_infos(self, context_infos: ContextInfos):
        self.context_infos.append(context_infos)

    def get_the_last_job_infos(self):
        if len(self.context_infos) == 0:
            return None
        return self.context_infos[-1].get_the_last_job_infos()

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.context_infos = self.context_infos[item]
            return self
        else:
            return self.context_infos[item]

    def __len__(self):
        return len(self.context_infos)

    def to_list(self):
        return [context_infos.to_dict() for context_infos in self.context_infos]

    @classmethod
    def from_list(cls, global_context_infos_list):
        context_infos = [ContextInfos.from_dict(context_infos_dict) for context_infos_dict in global_context_infos_list]
        return cls(context_infos)
