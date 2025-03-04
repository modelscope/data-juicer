# Example usage
from .generator import DuplicateGenerator, DuplicationConfig


def generate_benchmarks(base_dataset_path='data/c4_sample.jsonl'):
    """Generate benchmark datasets with various duplication configurations

    Args:
        base_dataset_path: Path to the base dataset file (JSONL format)
    """
    # 1. Basic usage with default settings
    generator = DuplicateGenerator()
    stats = generator.generate_from_dataset(
        dataset_path=base_dataset_path,
        output_path='output_with_duplicates.jsonl')

    # 2. Clustered duplicates with specific ratio
    #    clustered_config = DuplicationConfig(ratio=0.3,
    #                                         distribution='clustered',
    #                                         cluster_size=5,
    #                                         types={
    #                                             'exact': 0.2,
    #                                             'near': 0.6,
    #                                             'far': 0.2
    #                                         })

    # clustered_generator = DuplicateGenerator(clustered_config)
    # clustered_stats = clustered_generator.generate_from_dataset(
    #    dataset_path=base_dataset_path,
    #     output_path='output_clustered_duplicates.jsonl')

    # 3. High duplication rate with mostly near-duplicates
    # high_dup_config = DuplicationConfig(
    #    ratio=0.7,
    #    types={
    #        'exact': 0.1,
    #        'near': 0.8,
    #        'far': 0.1
    #    },
    #    modification_levels={
    #        'near': 0.05,
    #        'far': 0.2
    #    }  # Very subtle near-duplicates
    #    )

    # high_dup_generator = DuplicateGenerator(high_dup_config)
    # high_dup_stats = high_dup_generator.generate_from_dataset(
    #    dataset_path=base_dataset_path,
    #    output_path='output_high_duplication.jsonl')

    # 4. Generate benchmarks of different sizes
    for size_name, num_docs in [('small', 10000), ('medium', 100000),
                                ('large', 1000000)]:
        # Create sample of appropriate size
        sample_path = f'sample_{size_name}.jsonl'
        with open(sample_path, 'w') as outfile:
            with open(base_dataset_path, 'r') as infile:
                for i, line in enumerate(infile):
                    if i >= num_docs:
                        break
                    outfile.write(line)

        # Generate duplicates with different configurations
        for dup_rate in [0.1, 0.3, 0.5]:
            for dist in ['random', 'clustered']:
                config = DuplicationConfig(ratio=dup_rate, distribution=dist)
                generator = DuplicateGenerator(config)
                fn = f'output_{size_name}_{dist}_{int(dup_rate*100)}pct_dups.jsonl'  # noqa
                stats = generator.generate_from_dataset(
                    dataset_path=sample_path, output_path=fn)

                print(f'Generated {size_name} benchmark '
                      f'with {dist} distribution '
                      f'and {dup_rate*100}% duplicates')
                print(f'Stats: {stats}')


# If this file is run directly
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate benchmark datasets with controlled duplication')
    parser.add_argument('--dataset',
                        type=str,
                        default='data/c4_sample.jsonl',
                        help='Path to the base dataset file (JSONL format)')

    args = parser.parse_args()

    generate_benchmarks(args.dataset)
