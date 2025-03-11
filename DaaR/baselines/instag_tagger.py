import http.client
import json
import time
import os
from tqdm import tqdm

# Define cost table for models
cost_table = {
    "gpt-3.5-turbo": {"input_cost": 0.0035, "output_cost": 0.0105}
}

def call_api(combined_input):
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            conn = http.client.HTTPSConnection("[your api provider]")
            payload = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction:
[begin]
{combined_input}
[end]""".format(combined_input=combined_input)
                    },
                    {
                        "role": "user",
                        "content": """Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. Your response have to strictly follow this JSON format: [{"tag": str, "explanation": str}]. Please respond in English."""
                    }
                ]
            })
            headers = {
                'Authorization': 'Bearer [your api key]',
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            
            response_json = json.loads(data.decode("utf-8"))
            model_response = response_json['choices'][0]['message']['content']
            
            usage = response_json['usage']
            total_tokens = usage['total_tokens']
            
            model_name = "gpt-3.5-turbo"
            cost_info = cost_table.get(model_name, {"input_cost": 0, "output_cost": 0})
            input_cost = (usage['prompt_tokens'] / 1000) * cost_info["input_cost"]
            output_cost = (usage['completion_tokens'] / 1000) * cost_info["output_cost"]
            total_cost = input_cost + output_cost
            
            return model_response, total_tokens, total_cost
        
        except http.client.IncompleteRead as e:
            print(f"IncompleteRead error encountered. Retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(2)  # Wait for 2 seconds before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            time.sleep(2)  # Wait for 2 seconds before retrying
    
    raise Exception("Failed to complete API request after multiple retries.")

def process_data(input_file, output_file, num_samples=10000, print_interval=100):
    total_cost = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(tqdm(infile, total=num_samples)):
            if i >= num_samples:
                break
            
            data = json.loads(line.strip())
            instruction = data.get("instruction", "")
            input_data = data.get("input", "")
            
            # Concatenate instruction and input
            combined_input = f"{instruction}\nInput: {input_data}"
            
            try:
                evaluation, total_tokens, cost = call_api(combined_input)
                total_cost += cost
                
                processed_entry = {
                    "instruction": instruction,
                    "input": input_data,
                    "evaluation": evaluation,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
                
                # Write processed data to output file
                outfile.write(json.dumps(processed_entry) + '\n')
                outfile.flush()  # Ensure data is written immediately
                
                # Print cumulative cost in real time
                if (i + 1) % print_interval == 0:
                    print(f"\nCumulative Cost after {i + 1} samples: ${total_cost:.4f}")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")

# Example call
process_data("./data/raw/40k_data.jsonl", "./data/res/qw25/baselines/instag/with_score.jsonl", num_samples=40000)
