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
            try: 
                original_image = data.get("original_image", "")
                multihop_question = data.get("multihop_question") or data.get("multi_hop_question", "")
                multihop_answer = data.get("multihop_answer") or data.get("multi_hop_answer", "")
                thinking = data.get("thinking", "")

                if index % 4 == 0:  
                    if language == 'en':
                        messages = [
                            {"role": "user", "content": f"<image>" + "\nQuestion:" + multihop_question + "\nThinking process:" + thinking + "\nPlease generate the answer to this question based on the provided image, question and thinking process"},
                            {"role": "assistant", "content": "Answer:" + multihop_answer}
                        ]
                    elif language == 'cn':
                        messages = [
                            {"role": "user", "content": f"<image>" + "\n问题：" + multihop_question + "\n思考过程：" + thinking + "\n请根据提供的图片、问题和思考过程，生成这个问题的答案"},
                            {"role": "assistant", "content": "答案：" + multihop_answer}
                        ]
                    entries_0.append({
                        "messages": messages,
                        "images": [original_image]
                    })
                elif index % 4 == 1: 
                    if language == 'en':
                        messages = [
                            {"role": "user", "content": "<image>" + "\nQuestion:" + multihop_question + "\nAnswer:" + multihop_answer + "Please generate the thinking process to answer this question based on the provided image, question and answer"},
                            {"role": "assistant", "content": "Thinking process:" + thinking}
                        ]
                    elif language == 'cn':
                        messages = [
                            {"role": "user", "content": "<image>" + "\n问题：" + multihop_question + "\n答案：" + multihop_answer + "请根据提供的图片、问题和答案，生成解答这个问题的思维过程"},
                            {"role": "assistant", "content": "思考过程：" + thinking}
                        ]
                    entries_1.append({
                        "messages": messages,
                        "images": [original_image]
                    })
                elif index % 4 == 2:
                    if language == 'en':
                        messages = [
                            {"role": "user", "content": f"<image>" + "\nQuestion:" + multihop_question + "\nPlease generate the answer to this question and its thinking process based on the provided pictures and questions"},
                            {"role": "assistant", "content": "Answer:" + multihop_answer + "\nThinking process:" + thinking}
                        ]
                    elif language == 'cn':
                        messages = [
                            {"role": "user", "content": f"<image>" + "\n问题：" + multihop_question + "\n请根据提供的图片和问题，生成这个问题的答案及其思考过程"},
                            {"role": "assistant", "content": "答案：" + multihop_answer + "\n思考过程：" + thinking}
                        ]
                    entries_2.append({
                        "messages": messages,
                        "images": [original_image]
                    })
                elif index % 4 == 3:
                    if language == 'en':
                        messages = [
                            {"role": "user", "content": f"<image>" + "\nQuestion:" + multihop_question + "\nPlease generate the answer to this question based on the provided pictures and questions"},
                            {"role": "assistant", "content": "Answer:" + multihop_answer}
                        ]
                    elif language == 'cn':
                        messages = [
                            {"role": "user", "content": f"<image>" + "\n问题：" + multihop_question + "\n请根据提供的图片和问题，生成这个问题的答案"},
                            {"role": "assistant", "content": "答案：" + multihop_answer}
                        ]
                    entries_3.append({
                        "messages": messages,
                        "images": [original_image]
                    })
            except Exception as e: 
                print("出错了： ", e)
        
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
