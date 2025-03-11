import os
import json
import numpy as np
import math
from collections import Counter
import tqdm

def calculate_entropy(probabilities):
    """Calculate the entropy of a given probability distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def process_jsonl(input_file, output_file):
    """
    Process the JSONL file to add labels and entropy values to each record.
    
    Parameters:
    input_file (str): Path to the input JSONL file
    output_file (str): Path to the output JSONL file with labels and entropy
    """
    label_counter = Counter()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm.tqdm(infile, desc="Processing"):
            entry = json.loads(line.strip())
            
            # Get cluster_score and find the index of the max value as the label
            cluster_scores = entry.get("cluster_score", [])
            if not cluster_scores:
                raise ValueError("Cluster scores are missing or empty.")
            
            # Ensure cluster_score is a list of length 4
            if not isinstance(cluster_scores, list) or len(cluster_scores) != 4:
                raise ValueError("Invalid cluster_scores format.")

            # Find the index of the maximum score as the label
            label = int(np.argmax(cluster_scores))
            entry["label"] = label
            
            # Update label counter
            label_counter[label] += 1

            # Calculate entropy and add to entry
            entropy = calculate_entropy(cluster_scores)
            entry["entropy"] = entropy
            
            # Write updated entry to output file
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = '/home/daoyuan_dj/DaaR/daar/2_training/ce_res/qw25/infer_data_scores.jsonl'
    output_file = '/home/daoyuan_dj/DaaR/daar/2_training/ce_res/qw25/infer_data_entropy.jsonl'
    process_jsonl(input_file, output_file)
    print(f"Processed and saved results to {output_file}")
