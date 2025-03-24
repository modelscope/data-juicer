import json
import random
import os
import argparse
import re

def extract_number_from_filename(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else ''


def process_json(input_file, output_file_prefix, language):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)
        
        # Initialize four lists to store entries for each index % 4 case
        entries_0 = []
        entries_1 = []
        entries_2 = []
        entries_3 = []
        
        for index, data in enumerate(data_list):
            # Extracting components from the original json entry
            background_document = data.get("background_document", "")
            
            # Handle both possible field names for multihop_question
            multihop_question = data.get("multihop_question") or data.get("multi_hop_question", "")
            
            # Handle both possible field names for multihop_answer
            multihop_answer = data.get("multihop_answer") or data.get("multi_hop_answer", "")
            
            thinking = data.get("thinking", "")

            if index % 4 == 0:  
                if language == 'en':
                    entry1 = {
                        "instruction": "Below are a challenge question and the thinking process that assists in generating the answer. Please use this information to generate the multihop answer.",
                        "input": f"Question: {multihop_question}. Thinking: {thinking}",
                        "output": f"Anseer: {multihop_answer}."
                    }
                elif language == 'cn':
                    entry1 = {
                        "instruction": "以下是一个具有挑战性的问题以及帮助生成答案的思维过程。请使用这些信息来生成多跳答案。",
                        "input": f"问题：{multihop_question}。思维过程：{thinking}。",
                        "output": f"答案：{multihop_answer}"
                    }
                entries_0.append(entry1)
            elif index % 4 == 1: 
                if language == 'en':
                    entry1 = {
                        "instruction": "Below is a challenging question and its answer. Please generate a thought process to solve this question.",
                        "input": f"Question: {multihop_question}. Answer: {multihop_answer}.",
                        "output": f"Thinking: {thinking}"
                    }
                elif language == 'cn':
                    entry1 = {
                        "instruction": "下面是一个具有挑战性的问题及其答案。请生成解答这个问题的思维过程。",
                        "input": f"问题：{multihop_question}。答案：{multihop_answer}。",
                        "output": f"思考过程：{thinking}"
                    }
                entries_1.append(entry1)
            elif index % 4 == 2:
                if language == 'en':
                    entry1 = {
                        "instruction": "Below is a challenging question. Please generate an answer to this question and provide your thought process.",
                        "input": f"Question: {multihop_question}",
                        "output": f"Answer: {multihop_answer}. Thinking: {thinking}"
                    }
                elif language == 'cn':
                    entry1 = {
                        "instruction": "下面是一个具有挑战性的问题。请生成这个问题的答案，并提供思维过程。",
                        "input": f"问题：{multihop_question}",
                        "output": f"答案：{multihop_answer}，思考过程：{thinking}"
                    }
                entries_2.append(entry1)
            elif index % 4 == 3:
                if language == 'en':
                    entry1 = {
                        "instruction": "Below is a challenging question. Please generate an answer to this question.",
                        "input": f"Question: {multihop_question}",
                        "output": f"Answer: {multihop_answer}."
                    }
                elif language == 'cn':
                    entry1 = {
                        "instruction": "下面是一个具有挑战性的问题。请生成这个问题的答案。",
                        "input": f"问题：{multihop_question}",
                        "output": f"答案：{multihop_answer}"
                    }
                entries_3.append(entry1)
        
        # Write the accumulated entries to separate output files
        for i, entries in enumerate([entries_0, entries_1, entries_2, entries_3]):
            output_file = f"{output_file_prefix}_{i}.json"
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(entries, outfile, ensure_ascii=False, indent=4)


def process_output_field(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)
    
    for entry in data_list:
        output = entry.get("output")
        if isinstance(output, dict):
            # Convert the dict to a JSON string
            entry["output"] = json.dumps(output, ensure_ascii=False, indent=4)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data_list, outfile, ensure_ascii=False, indent=4)

def mix_and_shuffle_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
        
        combined_data = data1 + data2
        random.seed(42)  # Set random seed for reproducibility
        random.shuffle(combined_data)
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(combined_data, outfile, ensure_ascii=False, indent=4)

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description="Process JSON files for multilingual data.")
    parser.add_argument('--input_en_path', type=str, required=True, help='Path to the English input JSON file')
    parser.add_argument('--input_cn_path', type=str, required=True, help='Path to the Chinese input JSON file')
    args = parser.parse_args()

    # Extract numbers from input file names to construct output file names
    en_number = extract_number_from_filename(args.input_en_path)
    cn_number = extract_number_from_filename(args.input_cn_path)

    # Define output file paths based on extracted numbers
    output_file_en_prefix = f'./res/en_with_{en_number}'
    output_file_cn_prefix = f'./res/cn_with_{cn_number}'
    mix_output_file = f'./res/mix_with_{int(en_number) + int(cn_number)}.json'
    
    # Ensure output directories exist
    ensure_directory_exists(output_file_en_prefix)
    ensure_directory_exists(output_file_cn_prefix)
    ensure_directory_exists(mix_output_file)
    
    # Process and clean the English data
    process_json(args.input_en_path, output_file_en_prefix, 'en')
    for i in range(4):
        process_output_field(f"{output_file_en_prefix}_{i}.json", f"{output_file_en_prefix}_{i}.json")
    
    # Process and clean the Chinese data
    process_json(args.input_cn_path, output_file_cn_prefix, 'cn')
    for i in range(4):
        process_output_field(f"{output_file_cn_prefix}_{i}.json", f"{output_file_cn_prefix}_{i}.json")
    
    # # Mix and shuffle the data
    # mix_and_shuffle_files(output_file_en, output_file_cn, mix_output_file)

if __name__ == "__main__":
    main()
