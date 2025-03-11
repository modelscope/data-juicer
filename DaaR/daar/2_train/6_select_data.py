import os
import json
import argparse

def process_top_20_percent(input_file, output_file):

    # Read JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile:
        jsonl_entries = [json.loads(line.strip()) for line in infile]

    # Sort entries by predicted_entropy in descending order
    sorted_entries = sorted(jsonl_entries, key=lambda x: x['predicted_entropy'], reverse=True)
    
    # Calculate the number of top entries to select (20% of total)
    top_20_percent_count = int(len(sorted_entries) * 0.2)
    
    # Select the top 20% entries
    top_20_percent_entries = sorted_entries[:top_20_percent_count]

    # Write the top 20% entries to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in top_20_percent_entries:
            entry.pop('predicted_entropy', None)
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully processed and saved the top 20% entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process JSONL data to extract the top 20 entries based on predicted_entropy.")
    parser.add_argument('--input_file', type=str, default='./daar/2_training/mse_res/qw25/40k_infer_data.jsonl', help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, default='./daar/2_training/mse_res/qw25/daar_data.jsonl', help='Path to the output JSONL file.')
    args = parser.parse_args()

    process_top_20_percent(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
