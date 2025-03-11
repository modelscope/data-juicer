import os
import json
import argparse

def count_tokens(text):
    """Counts the number of tokens in a text based on spaces."""
    return len(text.split())

def process_data(input_file_path, output_path):
    """Processes the JSONL file and extracts the longest records based on instruction+input and output length."""
    # Read raw data
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Compute token count for instruction + input
    instruction_length_data = [(i, count_tokens(record['instruction'] + ' ' + record['input'])) for i, record in enumerate(data)]
    sorted_instruction_length_data = sorted(instruction_length_data, key=lambda x: x[1], reverse=True)[:8000]
    selected_instruction_records = [data[i] for i, _ in sorted_instruction_length_data]

    # Save instruction length data
    instruction_length_file = os.path.join(output_path, 'instruction_length.jsonl')
    with open(instruction_length_file, 'w', encoding='utf-8') as f:
        for record in selected_instruction_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Compute token count for output
    output_length_data = [(i, count_tokens(record['output'])) for i, record in enumerate(data)]
    sorted_output_length_data = sorted(output_length_data, key=lambda x: x[1], reverse=True)[:8000]
    selected_output_records = [data[i] for i, _ in sorted_output_length_data]

    # Save output length data
    output_length_file = os.path.join(output_path, 'output_length.jsonl')
    with open(output_length_file, 'w', encoding='utf-8') as f:
        for record in selected_output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("Data successfully processed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL data and filter the longest records.")
    parser.add_argument("--input_file_path", type=str, default="./data/raw/40k_data.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, default="./data/res/qw25/baselines/instruction_len", help="Directory to save the output JSONL files.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    process_data(args.input_file_path, args.output_path)