import hashlib
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from tqdm import tqdm

from .distributor import DuplicateDistributor
from .modifier import ModificationStrategy


@dataclass
class DuplicationConfig:
    """Configuration for duplication generation"""
    ratio: float = 0.3  # Percentage of duplicates in final dataset
    types: Dict[
        str,
        float] = None  # Distribution of duplicate types (exact, near, far)
    distribution: str = 'random'  # How duplicates are distributed
    cluster_size: int = 5  # Average size of duplicate clusters
    cluster_variance: float = 0.5  # Variance in cluster sizes
    modification_levels: Dict[str,
                              float] = None  # How much to modify for each type
    cross_source: bool = False  # Whether to create duplicates across sources

    def __post_init__(self):
        # Default duplicate type distribution
        if self.types is None:
            self.types = {'exact': 0.2, 'near': 0.5, 'far': 0.3}

        # Default modification levels
        if self.modification_levels is None:
            self.modification_levels = {
                'near': 0.1,  # 10% modification
                'far': 0.3  # 30% modification
            }

        # Validate configuration
        assert sum(self.types.values()
                   ) == 1.0, 'Duplicate type probabilities must sum to 1.0'
        assert 0 <= self.ratio <= 0.9, 'Dup ratio must be between 0 and 0.9'


class DuplicateGenerator:
    """Main class for generating controlled duplicates"""

    def __init__(self, config: DuplicationConfig = None):
        self.config = config or DuplicationConfig()

        # Initialize modification strategy
        self.modification_strategy = ModificationStrategy()

        # Use the distributor class instead of inline distribution logic
        self.distributor = DuplicateDistributor()

        # Map duplicate types to modifier methods
        self.modifiers = {
            'exact': self._modify_exact,
            'near': self._modify_near,
            'far': self._modify_far
        }

    def generate_from_dataset(self,
                              dataset_path: str,
                              output_path: str,
                              text_field: str = 'text',
                              id_field: str = 'id',
                              chunk_size: int = 100000,
                              num_processes: int = None) -> Dict[str, Any]:
        """Generate duplicates from an existing dataset file"""

        # Determine number of processes
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)

        # Count lines in file to determine total size
        total_lines = sum(1 for _ in open(dataset_path, 'r'))

        # Calculate number of chunks
        num_chunks = (total_lines + chunk_size - 1) // chunk_size

        # Process in chunks
        results = []
        for chunk_idx in range(num_chunks):
            print(f'Processing chunk {chunk_idx+1}/{num_chunks}')

            # Read chunk of documents
            start_line = chunk_idx * chunk_size
            end_line = min(start_line + chunk_size, total_lines)

            documents = []
            with open(dataset_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if i >= end_line:
                        break
                    try:
                        doc = json.loads(line)
                        documents.append(doc)
                    except json.JSONDecodeError:
                        continue

            # Generate duplicates for this chunk
            chunk_result = self._process_chunk(
                documents,
                output_path=f'{output_path}.chunk{chunk_idx}',
                text_field=text_field,
                id_field=id_field,
                num_processes=num_processes)

            results.append(chunk_result)

        # Combine chunks if needed
        if num_chunks > 1:
            self._combine_chunks(output_path, num_chunks)

        # Aggregate statistics
        stats = self._aggregate_stats(results)

        # Save statistics
        with open(f'{output_path}.stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def _process_chunk(self, documents: List[Dict[str, Any]], output_path: str,
                       text_field: str, id_field: str,
                       num_processes: int) -> Dict[str, Any]:
        """Process a chunk of documents to generate duplicates"""

        # Calculate how many duplicates to create
        num_originals = len(documents)
        num_duplicates = int(num_originals * self.config.ratio /
                             (1 - self.config.ratio))

        # Use the distributor class instead of inline distribution logic
        if self.config.distribution == 'clustered':
            distribution = self.distributor.clustered_distribution(
                documents,
                num_duplicates,
                avg_cluster_size=self.config.cluster_size,
                variance=self.config.cluster_variance)
        elif self.config.distribution == 'power_law':
            distribution = self.distributor.power_law_distribution(
                documents,
                num_duplicates,
                exponent=2.0  # Could be configurable
            )
        else:  # "random"
            distribution = self.distributor.random_distribution(
                documents, num_duplicates)

        # Process distribution to create duplicates
        duplicates = []

        # Prepare arguments for parallel processing
        args = []
        for original, count in distribution:
            args.append((original, count, text_field, id_field))

        # Process in parallel
        with mp.Pool(num_processes) as pool:
            duplicate_clusters = list(
                tqdm(pool.imap(self._generate_cluster, args),
                     total=len(args),
                     desc='Generating duplicate clusters'))

        # Flatten clusters
        for cluster in duplicate_clusters:
            duplicates.extend(cluster)

        # Add metadata to original documents
        for doc in documents:
            if 'is_duplicate' not in doc:
                doc['is_duplicate'] = False
                doc['original_id'] = None
                doc['duplicate_type'] = None

        # Combine and shuffle
        all_documents = documents + duplicates
        random.shuffle(all_documents)

        # Write to output file
        with open(output_path, 'w') as f:
            for doc in all_documents:
                f.write(json.dumps(doc) + '\n')

        # Return statistics
        return {
            'total_documents': len(all_documents),
            'original_documents': len(documents),
            'duplicate_documents': len(duplicates),
            'duplication_ratio': len(duplicates) / len(all_documents),
            'duplication_types': {
                dup_type: sum(1 for d in duplicates
                              if d.get('duplicate_type') == dup_type)
                for dup_type in self.config.types.keys()
            }
        }

    def _generate_cluster(self, args):
        """Generate a cluster of duplicates from a single original"""
        original, cluster_size, text_field, id_field = args

        duplicates = []
        for _ in range(cluster_size):
            dup_type = self._select_duplicate_type()
            duplicate = self._generate_duplicate(
                (original, dup_type, text_field, id_field))
            duplicates.append(duplicate)

        return duplicates

    def _generate_duplicate(self, args):
        """Generate a single duplicate from an original document"""
        original, dup_type, text_field, id_field = args

        # Create base duplicate
        duplicate = original.copy()

        # Add metadata
        duplicate['is_duplicate'] = True
        duplicate['original_id'] = original.get(id_field, 'unknown')
        duplicate['duplicate_type'] = dup_type

        # Generate new ID
        duplicate[id_field] = hashlib.md5(
            (str(original.get(id_field, '')) +
             str(random.random())).encode()).hexdigest()

        # Apply modifications based on duplicate type
        if text_field in duplicate:
            modifier_func = self.modifiers.get(dup_type, self._modify_exact)
            duplicate[text_field] = modifier_func(duplicate[text_field])

        return duplicate

    def _select_duplicate_type(self):
        """Select a duplicate type based on configured probabilities"""
        types, probs = zip(*self.config.types.items())
        return random.choices(types, weights=probs, k=1)[0]

    def _modify_exact(self, text):
        """No modification for exact duplicates"""
        return text

    def _modify_near(self, text):
        """Apply near-duplicate modifications using ModificationStrategy"""
        # Get modification intensity from config
        intensity = self.config.modification_levels.get('near', 0.1)

        # Use the ModificationStrategy to apply various modifications
        return self.modification_strategy.apply(text, intensity=intensity)

    def _modify_far(self, text):
        """Apply far-duplicate modifications using ModificationStrategy"""
        # Get modification intensity from config
        intensity = self.config.modification_levels.get('far', 0.3)

        # For far duplicates, we might want to apply more aggressive m
        # odifications
        # First shuffle paragraphs
        paragraphs = text.split('\n')
        if len(paragraphs) > 1:
            random.shuffle(paragraphs)

        # Then apply modifications to each paragraph
        modified_paragraphs = []
        for paragraph in paragraphs:
            # Apply the modification strategy with higher intensity
            modified = self.modification_strategy.apply(paragraph,
                                                        intensity=intensity)
            modified_paragraphs.append(modified)

        return '\n'.join(modified_paragraphs)

    def _get_similar_word(self, word):
        """Generate a similar word (placeholder implementation)"""
        # In a real implementation, you might use:
        # - Word embeddings to find semantically similar words
        # - Character-level modifications
        # - Synonym lookup

        # Simple implementation: modify the word slightly
        if len(word) <= 3:
            return word

        if random.random() < 0.5:
            # Change a character
            pos = random.randint(0, len(word) - 1)
            chars = list(word)
            chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
            return ''.join(chars)
        else:
            # Add or remove a character
            if random.random() < 0.5 and len(word) > 3:
                # Remove
                pos = random.randint(0, len(word) - 1)
                return word[:pos] + word[pos + 1:]
            else:
                # Add
                pos = random.randint(0, len(word))
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                return word[:pos] + char + word[pos:]

    def _combine_chunks(self, output_path, num_chunks):
        """Combine chunk files into a single output file"""
        with open(output_path, 'w') as outfile:
            for chunk_idx in range(num_chunks):
                chunk_path = f'{output_path}.chunk{chunk_idx}'
                with open(chunk_path, 'r') as infile:
                    for line in infile:
                        outfile.write(line)

                # Remove chunk file
                os.remove(chunk_path)

    def _aggregate_stats(self, chunk_stats):
        """Aggregate statistics from multiple chunks"""
        total_stats = {
            'total_documents': 0,
            'original_documents': 0,
            'duplicate_documents': 0,
            'duplication_types':
            {dup_type: 0
             for dup_type in self.config.types.keys()}
        }

        for stats in chunk_stats:
            total_stats['total_documents'] += stats['total_documents']
            total_stats['original_documents'] += stats['original_documents']
            total_stats['duplicate_documents'] += stats['duplicate_documents']

            for dup_type, count in stats['duplication_types'].items():
                total_stats['duplication_types'][dup_type] += count

        total_stats['duplication_ratio'] = (
            total_stats['duplicate_documents'] / total_stats['total_documents']
            if total_stats['total_documents'] > 0 else 0)

        return total_stats
