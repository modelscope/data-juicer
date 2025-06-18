import concurrent.futures
from typing import List

import numpy as np

from data_juicer.ops.base_op import Filter, Mapper, load_ops
from data_juicer.utils.constant import Fields
from data_juicer.utils.registry import Registry

OPERATORS = Registry('operators')


@OPERATORS.register_module('fused_filter')
class FusedFilter(Filter):
    """A fused operator for filters that can execute multiple filters in one pass."""

    _batched_op = True

    def __init__(self, name: str, fused_filters: List[Filter]):
        """Initialize the fused filter.

        Args:
            name: Name of the fused filter
            fused_filters: List of filters to be fused
        """
        self._name = name
        super().__init__()
        self.fused_filters = fused_filters

        # Set accelerator to 'cuda' if any of the fused filters use CUDA
        accelerator_methods = set(
            [op.accelerator for op in self.fused_filters])
        if 'cuda' in accelerator_methods:
            self.accelerator = 'cuda'

        # Update num_proc with the minimum of all fused filters
        self.num_proc = min([op.runtime_np() for op in self.fused_filters])

        # Store original operation configs
        self._op_cfg = {name: [op._op_cfg for op in self.fused_filters]}

        # Analyze filter dependencies
        self._analyze_dependencies()

        # Pre-allocate result arrays
        self._result_cache = {}

    def _analyze_dependencies(self):
        """Analyze dependencies between filters to optimize execution order."""
        # Create dependency graph
        self.dependency_graph = {}
        self.independent_groups = []

        for i, op1 in enumerate(self.fused_filters):
            self.dependency_graph[op1] = set()
            for j, op2 in enumerate(self.fused_filters):
                if i != j:
                    # Check if op2 depends on op1's output
                    if self._has_dependency(op1, op2):
                        self.dependency_graph[op1].add(op2)

        # Find independent groups
        visited = set()
        for op in self.fused_filters:
            if op not in visited:
                group = self._get_independent_group(op, visited)
                if group:
                    self.independent_groups.append(group)

    def _has_dependency(self, op1: Filter, op2: Filter) -> bool:
        """Check if op2 depends on op1's output."""
        # Get intermediate variables used by each operation
        op1_vars = set(op1._op_cfg.get('inter_vars', []))
        op2_vars = set(op2._op_cfg.get('inter_vars', []))

        # Check if op2 uses any variables produced by op1
        return bool(op1_vars & op2_vars)

    def _get_independent_group(self, start_op: Filter,
                               visited: set) -> List[Filter]:
        """Get a group of independent operations starting from start_op."""
        group = []
        to_visit = {start_op}

        while to_visit:
            op = to_visit.pop()
            if op not in visited:
                visited.add(op)
                group.append(op)
                # Add independent operations to visit
                for other_op in self.fused_filters:
                    if (other_op not in visited
                            and other_op not in self.dependency_graph[op]
                            and op not in self.dependency_graph[other_op]):
                        to_visit.add(other_op)

        return group

    def compute_stats_batched(self, samples, rank=None):
        """Compute statistics for all fused filters in one pass.

        Args:
            samples: Batch of samples to process
            rank: Rank for distributed processing

        Returns:
            Processed samples with computed statistics
        """
        import av

        # Initialize context for intermediate variables
        num_samples = len(samples[Fields.stats])
        samples[Fields.context] = [{} for _ in range(num_samples)]

        # Process independent groups in parallel
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_proc) as executor:
            futures = []
            for group in self.independent_groups:
                for op in group:
                    if op.accelerator == 'cuda':
                        futures.append(
                            executor.submit(op.compute_stats_batched,
                                            samples,
                                            rank=rank,
                                            context=True))
                    else:
                        futures.append(
                            executor.submit(op.compute_stats_batched,
                                            samples,
                                            context=True))

            # Wait for all operations to complete
            concurrent.futures.wait(futures)

        # Clean up contexts
        for ctx in samples[Fields.context]:
            for context_key in ctx:
                if isinstance(ctx[context_key], av.container.InputContainer):
                    ctx[context_key].streams.video[0].close()
                    ctx[context_key].close()

        # Remove context from samples
        _ = samples.pop(Fields.context)
        return samples

    def process_batched(self, samples):
        """Process samples through all fused filters.

        Args:
            samples: Batch of samples to process

        Returns:
            Boolean array indicating which samples pass all filters
        """
        # Initialize result array
        res = None

        # Process independent groups in parallel
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_proc) as executor:
            futures = []
            for group in self.independent_groups:
                group_futures = []
                for op in group:
                    future = executor.submit(op.process_batched, samples)
                    group_futures.append(future)
                futures.append(group_futures)

            # Process results in dependency order
            for group_futures in futures:
                group_results = []
                for future in group_futures:
                    this_res = np.array(list(future.result()))
                    group_results.append(this_res)

                # Combine results within group
                group_res = group_results[0]
                for this_res in group_results[1:]:
                    group_res = np.logical_and(group_res, this_res)

                # Combine with overall results
                if res is not None:
                    res = np.logical_and(res, group_res)
                else:
                    res = group_res

        return res

    def run(self, dataset, *, exporter=None, tracer=None):
        """Run the fused filter on a dataset.

        Args:
            dataset: Dataset to process
            exporter: Optional exporter for results
            tracer: Optional tracer for monitoring

        Returns:
            Processed dataset
        """
        # Prepare the dataset
        from data_juicer.core.data import NestedDataset
        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)

        # Initialize each filter
        for op in self.fused_filters:
            dataset = Filter.run(op, dataset)

        # Process the dataset
        new_dataset = dataset.map(
            self.process_batched,
            num_proc=self.num_proc,
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + '_process',
        )

        return new_dataset


@OPERATORS.register_module('fused_mapper')
class FusedMapper(Mapper):
    """A fused operator for mappers that can execute multiple mappers in one pass."""

    _batched_op = True

    def __init__(self,
                 name: str,
                 fused_mappers: List[str],
                 batch_size: int = 32):
        """Initialize the fused mapper.

        Args:
            name: Name of the fused mapper
            fused_mappers: List of mapper names to be fused
            batch_size: Batch size for processing
        """
        self._name = name
        super().__init__()
        self.batch_size = batch_size

        # Load the mapper operations
        self.fused_mappers = []
        for mapper_name in fused_mappers:
            mapper_config = {mapper_name: {}}
            mapper = load_ops([mapper_config])[0]
            self.fused_mappers.append(mapper)

        # Set accelerator to 'cuda' if any of the fused mappers use CUDA
        accelerator_methods = set(
            [op.accelerator for op in self.fused_mappers])
        if 'cuda' in accelerator_methods:
            self.accelerator = 'cuda'

        # Update num_proc with the minimum of all fused mappers
        self.num_proc = min([op.runtime_np() for op in self.fused_mappers])

        # Store original operation configs
        self._op_cfg = {name: [op._op_cfg for op in self.fused_mappers]}

    def process_batched(self, samples, rank=None):
        """Process samples through all fused mappers.

        Args:
            samples: Batch of samples to process
            rank: Rank for distributed processing

        Returns:
            Processed samples
        """
        # Process mappers sequentially
        for op in self.fused_mappers:
            process_args = {'rank': rank} if op.accelerator == 'cuda' else {}
            samples = op.process_batched(samples, **process_args)
        return samples

    def run(self, dataset, *, exporter=None, tracer=None):
        """Run the fused mapper on a dataset.

        Args:
            dataset: Dataset to process
            exporter: Optional exporter for results
            tracer: Optional tracer for monitoring

        Returns:
            Processed dataset
        """
        # Prepare the dataset
        from data_juicer.core.data import NestedDataset
        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)

        # Initialize each mapper
        for op in self.fused_mappers:
            dataset = Mapper.run(op, dataset)

        # Process the dataset
        new_dataset = dataset.map(
            self.process_batched,
            num_proc=self.num_proc,
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + '_process',
        )

        return new_dataset
