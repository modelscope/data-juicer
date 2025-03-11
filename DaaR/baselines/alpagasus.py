import http.client
import json
import time
import os
from tqdm import tqdm

# Cost Count
cost_table = {
    "gpt-3.5-turbo": {"input_cost": 0.0035, "output_cost": 0.0105}
}

def call_api(instruction, input_data, response):
    conn = http.client.HTTPSConnection("[your api provider]")
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": """We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.

Instruction: [Instruction]
Input: [Input]
Response: [Response]"""
            },
            {
                "role": "user",
                "content": f"""Please rate according to the accuracy of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of accuracy. The generated scores can be precise to decimal points, not just in increments of 0.5. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""
            }
        ]
    })
    headers = {
        'Authorization': 'Bearer [your api_key]',
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

def process_data(input_file, output_file, num_samples=200, print_interval=10):
    total_cost = 0
    processed_data = []

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(tqdm(infile, total=num_samples)):
            if i >= num_samples:
                break
            
            data = json.loads(line.strip())
            instruction = data.get("instruction", "")
            input_data = data.get("input", "")
            response_data = data.get("output", "")
            
            evaluation, total_tokens, cost = call_api(instruction, input_data, response_data)
            total_cost += cost
            
            processed_entry = {
                "instruction": instruction,
                "input": input_data,
                "output": response_data,
                "evaluation": evaluation,
                "total_tokens": total_tokens,
                "cost": cost
            }
            
            # Write processed data to output file
            outfile.write(json.dumps(processed_entry) + '\n')
            outfile.flush()  # Ensure data is written to file immediately
            
            # Print cumulative cost in real time
            if (i + 1) % print_interval == 0:
                print(f"\nCumulative Cost after {i + 1} samples: ${total_cost:.4f}")

# Example call
process_data("./data/raw/40k_data.jsonl", "./data/res/qw25/baselines/alpagasus/40k_alpagasus_score.jsonl", num_samples=40000, print_interval=1000)
