import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import json
import torch.nn.functional as F
import argparse
import tqdm

# Define CustomClassifier
class CustomClassifier(nn.Module):
    def __init__(self, base_model, input_dim, hidden_dim, output_dim=1):
        super(CustomClassifier, self).__init__()
        self.base_model = base_model
        layers = []
        layers.append(nn.Linear(input_dim, 2 * input_dim))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(2 * input_dim, input_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(input_dim, input_dim // 2))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(input_dim // 2, hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        if args.use_first_token:
            cls_embeddings = hidden_states[:, 0]
        elif args.use_mean_token:
            cls_embeddings = torch.mean(hidden_states, dim=1)
        else:
            cls_embeddings = hidden_states[:, -1]
        
        value = self.mlp(cls_embeddings)
        return value

# Args
parser = argparse.ArgumentParser(description='Infer Qwen2-7B Regressor')
parser.add_argument('--model_path', type=str, default="./daar/2_training/mse_res/qw25/mlp.pth", help='Path to the trained MLP model')
parser.add_argument('--base_model_path', type=str, default="./models/Qwen2.5-7B", help='Path to the base Qwen model')
parser.add_argument('--tokenizer_path', type=str, default="./models/Qwen2.5-7B", help='Path to the tokenizer')
parser.add_argument('--input_file', type=str, default="./data/raw/40k_data.jsonl", help='Input JSONL file path')
parser.add_argument('--clip_layer', type=int, default=3, help='Layer to clip')
parser.add_argument('--output_file', type=str, default="./daar/2_training/mse_res/qw25/40k_infer_data.jsonl", help='Output JSONL file path')
parser.add_argument('--use_first_token', action='store_true', help='Use the first token (similar to [CLS]) instead of the last token')
parser.add_argument('--use_mean_token', action='store_true', help='Use mean pooling of all tokens instead of the last token')
args = parser.parse_args()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, output_hidden_states=True)

# Clip LLM
num_layers_to_keep = args.clip_layer
base_model.model.layers = nn.ModuleList(base_model.model.layers[:num_layers_to_keep])

# Froze base model
for param in base_model.parameters():
    param.requires_grad = False

# Hidden size
input_dim = base_model.config.hidden_size
hidden_dim = 256
output_dim = 1

# Load MLP
model = CustomClassifier(base_model, input_dim, hidden_dim, output_dim).to(device)

checkpoint = torch.load(args.model_path, map_location=device)
state_dict = checkpoint['mlp_state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('mlp.', '')
    new_state_dict[name] = v

model.mlp.load_state_dict(new_state_dict, strict=False)
model.eval()

# Infer
def infer(model, tokenizer, input_text, device):
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=1024,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        value = model(input_ids=input_ids, attention_mask=attention_mask)
    
    return value.item()

# Read and write
with open(args.input_file, 'r') as f:
    data = [json.loads(line) for line in f]
    data = data[:]

# Data process
processed_data = []
for item in tqdm.tqdm(data, desc="Processing"):
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")
    combined_text = f"{instruction} {input_text} {output_text}"
    predicted_value = infer(model, tokenizer, combined_text, device)
    item["predicted_entropy"] = predicted_value
    processed_data.append(item)

# Save output
with open(args.output_file, 'w') as f:
    for item in processed_data:
        f.write(json.dumps(item) + '\n')

print(f"Processed and saved results to {args.output_file}")