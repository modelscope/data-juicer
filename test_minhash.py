import ray

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.ray_bts_minhash_deduplicator import \
    RayBTSMinhashDeduplicator

# Initialize Ray
ray.init()

# Create a sample dataset
ds_list = [
    {
        'text': 'Today is Sunday and it\'s a happy day!'
    },
    {
        'text': 'Do you need a cup of coffee?'
    },
    {
        'text':
        'Today is sunday and it\'s really a happy day!'  # Similar to first text
    },
    {
        'text': 'This paper proposed a novel method on LLM pretraining.'
    },
    {
        'text':
        'This paper proposed a novel method on LLM pretraining.'  # Duplicate
    }
]

# Create dataset and convert to Ray Dataset
dataset = Dataset.from_list(ds_list)
ray_dataset = ray.data.from_items(ds_list)

# Initialize the deduplicator with adjusted parameters
deduplicator = RayBTSMinhashDeduplicator(
    tokenization='space',  # For English text
    window_size=3,  # Smaller window to catch more similarities
    lowercase=True,
    ignore_pattern=r'\p{P}',  # Ignore punctuation
    jaccard_threshold=0.5,  # Lower threshold to catch more similar texts
    num_permutations=128,  # More permutations for better accuracy
    work_dir='./outputs/dedup_test'  # Output directory
)

# Run deduplication
result = deduplicator.run(ray_dataset)

# Convert result to list and print
result_list = result.take_all()
print('\nOriginal dataset size:', len(ds_list))
print('Deduplicated dataset size:', len(result_list))
print('\nDeduplicated texts:')
for item in result_list:
    print(f"- {item['text']}")

# Shutdown Ray
ray.shutdown()
