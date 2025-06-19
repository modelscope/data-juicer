import concurrent.futures
from typing import List

import numpy as np
from loguru import logger

from data_juicer.ops import load_ops
from data_juicer.ops.base_op import Filter, Mapper
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
            fused_filters: List of filters to fuse
        """
        super().__init__()
        self._name = name
        self.fused_filters = fused_filters

        # Add recursion prevention flag
        self._in_performance_test = False

        # Set accelerator based on available methods
        if any(hasattr(op, 'accelerator') and op.accelerator == 'cuda' for op in self.fused_filters):
            self.accelerator = 'cuda'
        else:
            self.accelerator = 'cpu'

        # Update num_proc with the minimum of all fused filters
        self.num_proc = min([op.runtime_np() for op in self.fused_filters])

        # Store original operation configs (create simple config if not available)
        self._op_cfg = {}
        for op in self.fused_filters:
            if hasattr(op, 'config') and op.config:
                self._op_cfg[op._name] = op.config
            else:
                # Create a simple config for filters without explicit config
                self._op_cfg[op._name] = {'inter_vars': [], 'dependencies': []}

        # Analyze dependencies and determine execution strategy
        self._analyze_dependencies()
        self._determine_execution_strategy()

        # Analyze filter dependencies
        self._analyze_dependencies()

        # Pre-allocate result arrays
        self._result_cache = {}

        # Log the chosen strategy
        logger.info(f"FusedFilter '{name}' using {self.execution_strategy} execution strategy")
        if self.has_dependencies:
            logger.info('  Reason: Filters have dependencies')
        else:
            simple_count = sum(
                1 for op in self.fused_filters if op._name in {
                    'text_length_filter', 'words_num_filter', 'character_repetition_filter', 'word_repetition_filter',
                    'special_characters_filter', 'alphanumeric_filter', 'average_line_length_filter',
                    'maximum_line_length_filter'
                })
            complex_count = len(self.fused_filters) - simple_count
            logger.info(f'  Reason: {simple_count} simple filters, {complex_count} complex filters')

    def _analyze_dependencies(self):
        """Analyze dependencies between filters to optimize execution order."""
        # Create dependency graph
        self.dependency_graph = {}
        self.independent_groups = []
        self.has_dependencies = False

        for i, op1 in enumerate(self.fused_filters):
            self.dependency_graph[op1] = set()
            for j, op2 in enumerate(self.fused_filters):
                if i != j:
                    # Check if op2 depends on op1's output
                    if self._has_dependency(op1, op2):
                        self.dependency_graph[op1].add(op2)
                        self.has_dependencies = True

        # Find independent groups
        visited = set()
        for op in self.fused_filters:
            if op not in visited:
                group = self._get_independent_group(op, visited)
                if group:
                    self.independent_groups.append(group)

        # Determine execution strategy
        self.execution_strategy = self._determine_execution_strategy()

    def _has_dependency(self, op1: Filter, op2: Filter) -> bool:
        """Check if op2 depends on op1's output."""
        # Get intermediate variables used by each operation from stored configs
        op1_vars = set(self._op_cfg.get(op1._name, {}).get('inter_vars', []))
        op2_vars = set(self._op_cfg.get(op2._name, {}).get('inter_vars', []))

        # Check if op2 uses any variables produced by op1
        return bool(op1_vars & op2_vars)

    def _get_independent_group(self, start_op: Filter, visited: set) -> List[Filter]:
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
                    if (other_op not in visited and other_op not in self.dependency_graph[op] and
                            op not in self.dependency_graph[other_op]):
                        to_visit.add(other_op)

        return group

    def _determine_execution_strategy(self):
        """Determine the best execution strategy based on filter characteristics."""
        if self.has_dependencies:
            return 'sequential'

        # Check if filters are simple enough for parallel execution
        simple_filters = 0
        complex_filters = 0

        for op in self.fused_filters:
            # Simple filters: text_length, words_num, character_repetition
            # Complex filters: perplexity, stopwords, flagged_words
            simple_filter_names = {
                'text_length_filter', 'words_num_filter', 'character_repetition_filter', 'word_repetition_filter',
                'special_characters_filter', 'alphanumeric_filter', 'average_line_length_filter',
                'maximum_line_length_filter'
            }

            if op._name in simple_filter_names:
                simple_filters += 1
            else:
                complex_filters += 1

        # Use parallel if mostly simple filters, sequential if complex filters
        if complex_filters > simple_filters:
            return 'sequential'
        else:
            return 'parallel'

    def _should_skip_fusion(self, sample_size: int = 1000) -> bool:
        """Determine if fusion should be skipped based on performance analysis.

        Args:
            sample_size: Number of samples being processed

        Returns:
            True if fusion should be skipped, False if fusion is beneficial
        """
        # Prevent recursion during performance testing
        if self._in_performance_test:
            return False

        # Skip performance test for very large datasets (fusion is always beneficial)
        if sample_size > 10000:
            return False

        # Skip performance test for complex filters (always use fusion)
        complex_filter_names = {
            'perplexity_filter', 'stopwords_filter', 'flagged_words_filter', 'language_id_score_filter',
            'word_repetition_filter'
        }
        complex_count = sum(1 for op in self.fused_filters if op._name in complex_filter_names)

        # Always use fusion for complex filters
        if complex_count > 0:
            return False

        # Skip fusion for single filters
        if len(self.fused_filters) == 1:
            return True

        # For simple filters, run a quick performance test
        try:
            return self._quick_performance_test(min(100, sample_size))
        except Exception as e:
            logger.warning(f'Performance test failed: {e}. Defaulting to fusion.')
            return False

    def _quick_performance_test(self, sample_size: int) -> bool:
        """Run a quick performance test to determine if fusion is beneficial.

        Args:
            sample_size: Number of samples to test (small sample for speed)

        Returns:
            True if fusion should be skipped, False if fusion is beneficial
        """
        import random
        import string
        import time

        from data_juicer.utils.constant import Fields

        # Set recursion prevention flag
        self._in_performance_test = True

        try:
            # Create minimal test data
            test_data = {
                'text': [''.join(random.choices(string.ascii_letters + ' ', k=50)) for _ in range(sample_size)],
                Fields.stats: [{} for _ in range(sample_size)]
            }

            # Measure individual execution time
            individual_start = time.time()
            for op in self.fused_filters:
                op.compute_stats_batched(test_data.copy())
                op.process_batched(test_data.copy())
            individual_time = time.time() - individual_start

            # Measure fused execution time
            fused_start = time.time()
            self.compute_stats_batched(test_data.copy())
            self.process_batched(test_data.copy())
            fused_time = time.time() - fused_start

            # Calculate overhead ratio
            overhead_ratio = fused_time / individual_time if individual_time > 0 else float('inf')

            # Decision logic for simple filters only
            if individual_time < 0.001:
                # Very fast filters - overhead not worth it
                should_skip = True
            elif overhead_ratio > 3.0 and individual_time < 0.01:
                # Simple filters with high overhead - skip fusion
                should_skip = True
            else:
                # Default to fusion for most cases
                should_skip = False

            logger.info(f'Performance test: individual={individual_time:.3f}s, '
                        f'fused={fused_time:.3f}s, ratio={overhead_ratio:.2f}x, '
                        f'skip={should_skip}')

            return should_skip

        finally:
            # Always clear the recursion prevention flag
            self._in_performance_test = False

    def compute_stats_batched(self, samples, rank=None):
        """Compute statistics for all fused filters using the best strategy."""
        import av

        # Check if we should skip fusion based on performance analysis
        if self._should_skip_fusion(len(samples[Fields.stats])):
            from loguru import logger
            logger.debug(f'Skipping fusion for {self._name} - executing filters individually')

            # Execute filters individually (no fusion)
            for op in self.fused_filters:
                if op.accelerator == 'cuda':
                    samples = op.compute_stats_batched(samples, rank=rank)
                else:
                    samples = op.compute_stats_batched(samples)
            return samples

        # Initialize context for intermediate variables
        num_samples = len(samples[Fields.stats])
        samples[Fields.context] = [{} for _ in range(num_samples)]

        if self.execution_strategy == 'parallel':
            # Parallel execution for independent filters
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                futures = []
                for group in self.independent_groups:
                    for op in group:
                        if op.accelerator == 'cuda':
                            futures.append(executor.submit(op.compute_stats_batched, samples, rank=rank, context=True))
                        else:
                            futures.append(executor.submit(op.compute_stats_batched, samples, context=True))

                # Wait for all operations to complete
                concurrent.futures.wait(futures)
        else:
            # Sequential execution for dependent or complex filters
            for op in self.fused_filters:
                if op.accelerator == 'cuda':
                    samples = op.compute_stats_batched(samples, rank=rank, context=True)
                else:
                    samples = op.compute_stats_batched(samples, context=True)

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
        """Process samples through all fused filters using the best strategy."""
        # Check if we should skip fusion based on performance analysis
        if self._should_skip_fusion(len(samples[Fields.stats])):
            from loguru import logger
            logger.debug(f'Skipping fusion for {self._name} - processing filters individually')

            # Process filters individually (no fusion)
            result = None
            for op in self.fused_filters:
                filter_result = list(op.process_batched(samples))

                if result is None:
                    result = filter_result
                else:
                    # Combine with logical AND (sample must pass all filters)
                    result = [r1 and r2 for r1, r2 in zip(result, filter_result)]

            return result

        if self.execution_strategy == 'parallel':
            # Parallel execution - all filters see original data
            return self._process_batched_parallel(samples)
        else:
            # Sequential execution - each filter sees previous filter's output
            return self._process_batched_sequential(samples)

    def _process_batched_parallel(self, samples):
        """Process filters in parallel (all see original data)."""
        # Initialize result array
        res = None

        # Process independent groups in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_proc) as executor:
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

    def _process_batched_sequential(self, samples):
        """Process filters sequentially (each sees previous output)."""
        # Process filters sequentially to match individual execution behavior
        result = None

        for op in self.fused_filters:
            filter_result = list(op.process_batched(samples))

            if result is None:
                result = filter_result
            else:
                # Combine with logical AND (sample must pass all filters)
                result = [r1 and r2 for r1, r2 in zip(result, filter_result)]

        return result

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

    def __init__(self, name: str, fused_mappers: List[str], batch_size: int = 32):
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
        accelerator_methods = set([op.accelerator for op in self.fused_mappers])
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
