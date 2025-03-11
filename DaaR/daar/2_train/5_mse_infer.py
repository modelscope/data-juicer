import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import json
import torch.nn.functional as F
import argparse
import tqdm

# 定义CustomClassifier类
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
        
        value = self.mlp(cls_embeddings)  # 输出连续值
        return value

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Infer Qwen2-7B Regressor')
parser.add_argument('--model_path', type=str, default="./daar/2_training/mse_res/qw25/mlp.pth", help='Path to the trained MLP model')
parser.add_argument('--base_model_path', type=str, default="./models/Qwen2.5-7B", help='Path to the base Qwen model')
parser.add_argument('--tokenizer_path', type=str, default="./models/Qwen2.5-7B", help='Path to the tokenizer')
parser.add_argument('--input_file', type=str, default="./data/raw/40k_data.jsonl", help='Input JSONL file path')
parser.add_argument('--output_file', type=str, default="./daar/2_training/mse_res/qw25/40k_infer_data.jsonl", help='Output JSONL file path')
parser.add_argument('--use_first_token', action='store_true', help='Use the first token (similar to [CLS]) instead of the last token')
parser.add_argument('--use_mean_token', action='store_true', help='Use mean pooling of all tokens instead of the last token')
args = parser.parse_args()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, output_hidden_states=True)

# 截断模型到第五层
num_layers_to_keep = 3
base_model.model.layers = nn.ModuleList(base_model.model.layers[:num_layers_to_keep])

# 冻结 base_model 的所有层
for param in base_model.parameters():
    param.requires_grad = False

# 获取模型配置的hidden size
input_dim = base_model.config.hidden_size
hidden_dim = 256
output_dim = 1  # 修改为 1

# 加载MLP模型
model = CustomClassifier(base_model, input_dim, hidden_dim, output_dim).to(device)  # 将模型移动到设备

checkpoint = torch.load(args.model_path, map_location=device)  # 指定加载位置为设备
state_dict = checkpoint['mlp_state_dict']

# 去掉键的前缀
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('mlp.', '')  # 去掉前缀 'mlp.'
    new_state_dict[name] = v

model.mlp.load_state_dict(new_state_dict, strict=False)
model.eval()

# 进行推理
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
    
    return value.item()  # 返回一个标量值

# 读取输入文件
with open(args.input_file, 'r') as f:
    data = [json.loads(line) for line in f]
    data = data[:]

# 处理数据
processed_data = []
for item in tqdm.tqdm(data, desc="Processing"):
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")
    combined_text = f"{instruction} {input_text} {output_text}"
    predicted_value = infer(model, tokenizer, combined_text, device)
    item["predicted_entropy"] = predicted_value
    processed_data.append(item)

# 保存输出文件
with open(args.output_file, 'w') as f:
    for item in processed_data:
        f.write(json.dumps(item) + '\n')

print(f"Processed and saved results to {args.output_file}")