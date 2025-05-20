import os
import json
import argparse

def process_overall_selection(input_file, output_file):
    # Select data by sorted predicted_entropy
    with open(input_file, 'r', encoding='utf-8') as infile:
        entries = [json.loads(line.strip()) for line in infile]

    sorted_entries = sorted(entries, key=lambda x: x['predicted_entropy'], reverse=True)
    top_20_percent_count = int(len(sorted_entries) * 0.2)
    selected_entries = sorted_entries[:top_20_percent_count]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in selected_entries:
            entry.pop('predicted_entropy', None)
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Save to {output_file}")

def process_customized_selection(input_file, output_file, customized_domain):
    # Select data considering pseudo_label
    with open(input_file, 'r', encoding='utf-8') as infile:
        entries = [json.loads(line.strip()) for line in infile]

    # Categorize domains
    domain_entries = {
        'coding': [],
        'common_sense': [],
        'reasoning': [],
        'mathematics': []
    }

    for entry in entries:
        domain = entry.get('pseudo_label')
        if domain in domain_entries:
            domain_entries[domain].append(entry)

    selected_entries = []

    # Process on each domain
    for domain in domain_entries:
        entries_group = domain_entries[domain]
        sorted_group = sorted(entries_group, key=lambda x: x['predicted_entropy'], reverse=True)
        if domain == customized_domain:
            count = int(len(sorted_group) * 0.5)
        else:
            count = int(len(sorted_group) * 0.1)
        selected_entries.extend(sorted_group[:count])

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in selected_entries:
            entry.pop('predicted_entropy', None)
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("Sucessfully process.")

def main():
    parser = argparse.ArgumentParser(description="Select data by DaaR")
    parser.add_argument('--input_file', type=str, default='./daar/2_training/mse_res/qw3/40k_infer_data.jsonl', help='Input jsonl path')
    parser.add_argument('--output_file', type=str, default='./daar/2_training/mse_res/qw3/daar_data.jsonl', help='Input jsonl path')
    parser.add_argument('--selection_strategy', type=str, default='overall', choices=['overall', 'customized'], help='Selection strategy')
    parser.add_argument('--customized_domain', type=str, default=None, choices=['coding', 'common_sense', 'reasoning', 'mathematics'], help='Customized domain')

    args = parser.parse_args()

    # Process selection
    if args.selection_strategy == 'customized':
        process_customized_selection(args.input_file, args.output_file, args.customized_domain)
    else:
        process_overall_selection(args.input_file, args.output_file)

if __name__ == "__main__":
    main()