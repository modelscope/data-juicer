import os
import shutil
import time
from typing import Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.adapter import Adapter
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


class TempDirManager:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def __enter__(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.tmp_dir):
            logger.info(f"Removing tmp dir {self.tmp_dir} ...")
            shutil.rmtree(self.tmp_dir)


class RayExecutor(ExecutorBase, EventLoggingMixin, DAGExecutionMixin):
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions such as checkpoint, tracer are not supported.

    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        super().__init__(cfg)

        self.executor_type = "ray"
        self.work_dir = self.cfg.work_dir

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)
        self.adapter = Adapter(self.cfg)

        # init ray
        logger.info("Initializing Ray ...")
        ray.init(self.cfg.ray_address)
        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # absolute path resolution logic

        # init dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        logger.info("Preparing exporter...")
        self.exporter = RayExporter(
            self.cfg.export_path,
            keep_stats_in_res_ds=self.cfg.keep_stats_in_res_ds,
            keep_hashes_in_res_ds=self.cfg.keep_hashes_in_res_ds,
        )

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Running the dataset process pipeline

        :param load_data_np: number of workers when loading the dataset.
        :param skip_return: skip return for API called.
        :return: processed dataset.
        """
        # 1. load data
        logger.info("Loading dataset with Ray...")
        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np)
        columns = dataset.schema().columns

        # 2. extract processes
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process)

        # Initialize DAG execution planning
        self._initialize_dag_execution(self.cfg)

        # Log job start with DAG context
        job_config = {
            "dataset_path": self.cfg.dataset_path,
            "work_dir": self.work_dir,
            "executor_type": self.executor_type,
            "dag_node_count": len(self.pipeline_dag.nodes) if self.pipeline_dag else 0,
            "dag_edge_count": len(self.pipeline_dag.edges) if self.pipeline_dag else 0,
            "parallel_groups_count": len(self.pipeline_dag.parallel_groups) if self.pipeline_dag else 0,
        }
        self.log_job_start(job_config, len(ops))

        if self.cfg.op_fusion:
            probe_res = None
            if self.cfg.fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                probe_res, _ = self.adapter.probe_small_batch(dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy " f"[{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        with TempDirManager(self.tmp_dir):
            # 3. data process with DAG monitoring
            logger.info("Processing data with DAG monitoring...")
            tstart = time.time()

            # Use DAG-aware execution if available
            if self.pipeline_dag:
                self._execute_operations_with_dag_monitoring(dataset, ops)
            else:
                # Fallback to normal execution
                dataset.process(ops)

            # 4. data export
            logger.info("Exporting dataset to disk...")
            self.exporter.export(dataset.data, columns=columns)
            tend = time.time()
            logger.info(f"All Ops are done in {tend - tstart:.3f}s.")

        # Log job completion with DAG context
        job_duration = time.time() - tstart
        self.log_job_complete(job_duration, self.cfg.export_path)

        if not skip_return:
            return dataset
