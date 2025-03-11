import jsonlines
from collections import defaultdict
from tqdm import tqdm
import random

# Step 1: Load and pair two files
def load_and_pair_files(tag_file_path, data_file_path):
    tag_data_pairs = []
    
    # Load tag information
    with jsonlines.open(tag_file_path) as tag_reader:
        tags = [obj for obj in tag_reader]
    
    # Load raw data
    with jsonlines.open(data_file_path) as data_reader:
        data = [obj for obj in data_reader]
    
    # Pair tags with raw data
    for tag, datum in zip(tags, data):
        tag_data_pairs.append((tag, datum))
    
    return tag_data_pairs

# Step 2: Data selection based on complexity (unchanged)
def sample_data_for_complexity(tag_data_pairs, subset_size=8000, batch_size=100):
    complexity_subsets = []

    # Sort the data pool in descending order by the number of tags
    remaining_pool = sorted(tag_data_pairs, key=lambda x: len(x[0]['formatted_tags'].split('\n')), reverse=True)

    while len(complexity_subsets) < subset_size and remaining_pool:
        current_subset = []
        current_tags = set()

        for _ in tqdm(range(batch_size), desc="Sampling for Complexity", ncols=100):
            if not remaining_pool:
                break
            found = False
            for i, (tag_info, datum) in enumerate(remaining_pool):
                new_tags = {tag.split(': ')[-1].strip().lower() for tag in tag_info['formatted_tags'].split('\n')[1:]}
                if not current_tags.isdisjoint(new_tags):  
                    current_subset.append(datum)
                    current_tags.update(new_tags)
                    del remaining_pool[i]  
                    found = True
                    break
            
            if not found:  
                tag_info, datum = remaining_pool.pop(random.randrange(len(remaining_pool)))
                current_subset.append(datum)
                current_tags.update({tag.split(': ')[-1].strip().lower() for tag in tag_info['formatted_tags'].split('\n')[1:]})

        complexity_subsets.extend(current_subset)

    return complexity_subsets[:subset_size]

# Step 3: Data selection based on diversity
def sample_data_for_diversity(tag_data_pairs, subset_size=8000, coverage_rate=0.8):
    diversity_subsets = []
    remaining_pool = list(tag_data_pairs)
    global_tag_set = set()
    all_possible_tags = set()

    # First, collect all possible tags
    for tag_info, _ in remaining_pool:
        all_possible_tags.update({tag.split(': ')[-1].strip().lower() for tag in tag_info['formatted_tags'].split('\n')[1:]})
    
    # Adjust coverage rate if there are not enough tags to meet the requirement
    if not all_possible_tags:
        coverage_rate = 0
    else:
        coverage_rate = min(coverage_rate, len(global_tag_set) / len(all_possible_tags))

    while len(diversity_subsets) < subset_size and remaining_pool:
        best_item = None
        best_coverage_increase = 0
        
        # Temporarily store the current best data pair that increases coverage
        for tag_info, datum in tqdm(remaining_pool, desc="Sampling for Diversity", ncols=100):
            new_tags = {tag.split(': ')[-1].strip().lower() for tag in tag_info['formatted_tags'].split('\n')[1:]}
            updated_global_tag_set = global_tag_set.union(new_tags)
            
            # Calculate coverage increase
            if all_possible_tags:
                coverage_increase = (len(updated_global_tag_set) - len(global_tag_set)) / len(all_possible_tags)
            else:
                coverage_increase = 0

            if coverage_increase > best_coverage_increase:
                best_coverage_increase = coverage_increase
                best_item = (tag_info, datum)

        if best_item is not None:
            tag_info, datum = best_item
            diversity_subsets.append(datum)
            global_tag_set.update({tag.split(': ')[-1].strip().lower() for tag in tag_info['formatted_tags'].split('\n')[1:]})
            remaining_pool.remove(best_item)

            # Check if the preset coverage rate is reached
            current_coverage = len(global_tag_set) / len(all_possible_tags) if all_possible_tags else 0
            print(f"Current Coverage: {current_coverage:.4f}")  # Print current coverage for debugging

            if current_coverage >= coverage_rate:
                print("Coverage rate reached. Stopping early.")
                break

    return diversity_subsets[:subset_size]

# Load and pair files
tag_file_path = './data/res/qw25/baselines/instag/tags.jsonl'
data_file_path = './data/raw/40k_data.jsonl'
tag_data_pairs = load_and_pair_files(tag_file_path, data_file_path)

# Data selection based on complexity
complexity_sampled_data = sample_data_for_complexity(tag_data_pairs)

# Data selection based on diversity
diversity_sampled_data = sample_data_for_diversity(tag_data_pairs)

# Save data selected based on complexity
with jsonlines.open('./data/res/qw25/baselines/instag/instag_c.jsonl', mode='w') as writer:
    writer.write_all(complexity_sampled_data)

# Save data selected based on diversity
with jsonlines.open('./data/res/qw25/baselines/instag/instag_d.jsonl', mode='w') as writer:
    writer.write_all(diversity_sampled_data)
