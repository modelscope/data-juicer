import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import json
import numpy as np
import tqdm
import os
import argparse
import matplotlib.pyplot as plt

# Dataset definition
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output_text = item["output"]
        label = item["label"] 
        
        # Concat Input and Output
        text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
        
        # Text embedding
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Input_ids and mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

# Definition of MLP structure
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        layers = [
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ]
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

# args
parser = argparse.ArgumentParser(description='Train Qwen2-7B Classifier')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--log_save', type=float, default=50, help='Log Save')
parser.add_argument('--eval_step', type=float, default=200, help='Eval step')
parser.add_argument('--output_dir', type=str, default='./daar/2_training/ce_res/qw25', help='Output directory for results')
parser.add_argument('--use_first_token', action='store_true', help='Use the first token (similar to [CLS]) instead of the last token')
parser.add_argument('--use_mean_pooling', action='store_true', help='Use mean pooling of all tokens instead of the last token')
parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-7B', help='Model path')
parser.add_argument('--file_path', type=str, default='./daar/1_centroid/train_data/qw25/train_data.jsonl', help='Used file path')
args = parser.parse_args()

# Output file
os.makedirs(args.output_dir, exist_ok=True)

log_save = args.log_save
eval_step = args.eval_step

# Pretrain and tokenizer
model_name = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Setting pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Clip LLM to layer-5
num_layers_to_keep = 3
model.model.layers = nn.ModuleList(model.model.layers[:num_layers_to_keep])

# Get last layer dimension
last_hidden_state_dim = model.config.hidden_size

# MLP configs
hidden_dim = 256
output_dim = 4

# Init MLP
mlp = MLP(last_hidden_state_dim, hidden_dim, output_dim)

# Freeze pretrain-LLM
for param in model.parameters():
    param.requires_grad = False

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
mlp.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

# Dataset load
file_path = args.file_path
dataset = CustomDataset(file_path, tokenizer)

# Split to train and eval
train_size = int(0.97 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    n_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            if args.use_first_token:
                cls_embeddings = hidden_states[:, 0]
            elif args.use_mean_pooling:
                cls_embeddings = torch.mean(hidden_states, dim=1)
            else:
                cls_embeddings = hidden_states[:, -1]

            logits = mlp(cls_embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            n_examples += labels.size(0)

    return correct_predictions.double() / n_examples, total_loss / len(dataloader)

# Train and Validate
def train_and_validate_mlp(train_dataloader, val_dataloader, mlp, criterion, optimizer, device, num_epochs=1):
    train_losses = []
    val_losses = []
    val_accuracies = []
    log_file = os.path.join(args.output_dir, 'training_log.txt')

    step = 0
    for epoch in range(num_epochs):
        # Self-check
        for name, param in model.named_parameters():
            if param.requires_grad:
                raise ValueError(f"Parameter {name} is not frozen. Ensure all parameters of the base model are frozen.")

        mlp.train()
        running_train_loss = 0.0
        correct_predictions = 0
        n_examples = 0

        for batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Last layer output
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # Self-check
                expected_hidden_states = num_layers_to_keep + 1
                if len(hidden_states) != expected_hidden_states:
                    raise ValueError(f"Expected {expected_hidden_states} hidden states, but got {len(hidden_states)}. Check layer truncation.")
                
                hidden_states = hidden_states[-1]

            if args.use_first_token:
                cls_embeddings = hidden_states[:, 0]
            elif args.use_mean_pooling:
                cls_embeddings = torch.mean(hidden_states, dim=1)
            else:
                cls_embeddings = hidden_states[:, -1]

            # FP
            logits = mlp(cls_embeddings)
            loss = criterion(logits, labels)

            # BP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            n_examples += labels.size(0)

            step += 1

            # Training Loss
            if step % log_save == 0:
                avg_train_loss = running_train_loss / 20
                print(f'Step {step}: Train Loss {avg_train_loss}')
                train_losses.append((step, avg_train_loss))
                running_train_loss = 0.0
                with open(log_file, 'a') as f:
                    f.write(f'Step {step}: Train Loss {avg_train_loss}\n')

            # Validate Step
            if step % eval_step == 0:
                val_acc, val_loss = eval_model(model, val_dataloader, criterion, device)
                print(f'Step {step}: Val Loss {val_loss} Accuracy {val_acc}')
                val_losses.append((step, val_loss))
                val_accuracies.append((step, val_acc.cpu().item()))
                with open(log_file, 'a') as f:
                    f.write(f'Step {step}: Val Loss {val_loss} Accuracy {val_acc}\n')

    # Plot train curve
    plt.figure(figsize=(12, 6))
    steps, losses = zip(*train_losses)
    plt.plot(steps, losses, label='Train Loss', marker='o')
    for s, l in zip(steps, losses):
        plt.annotate(f'{l:.3f}', xy=(s, l), xytext=(s, l + 0.01), ha='center')
    plt.title('Training Loss over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'train_loss_curve.png'))
    plt.show()

    # Plot eval curve
    plt.figure(figsize=(12, 6))
    val_steps, val_losses_vals = zip(*val_losses)
    val_steps_acc, val_accuracies_vals = zip(*val_accuracies)
    plt.plot(val_steps, val_losses_vals, label='Validation Loss', marker='x')
    for s, l in zip(val_steps, val_losses_vals):
        plt.annotate(f'{l:.3f}', xy=(s, l), xytext=(s, l + 0.01), ha='center')
    plt.plot(val_steps_acc, val_accuracies_vals, label='Validation Accuracy', marker='o')
    for s, a in zip(val_steps_acc, val_accuracies_vals):
        plt.annotate(f'{a:.3f}', xy=(s, a), xytext=(s, a + 0.01), ha='center')
    plt.title('Validation Loss and Accuracy over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'validation_metrics_curve.png'))
    plt.show()

    # Save MLP
    mlp_save_path = os.path.join(args.output_dir, 'mlp.pth')
    os.makedirs(os.path.dirname(mlp_save_path), exist_ok=True)
    torch.save({'mlp_state_dict': mlp.state_dict()}, mlp_save_path)
    print(f'MLP layer saved to {mlp_save_path}')

# Self-check
for name, param in model.named_parameters():
    if param.requires_grad:
        raise ValueError(f"Parameter {name} is not frozen. Ensure all parameters of the base model are frozen.")

# Train start
train_and_validate_mlp(train_dataloader, val_dataloader, mlp, criterion, optimizer, device)

# A predict
def predict(text, model, tokenizer, mlp, device, use_first_token=False, use_mean_pooling=False):
    model.eval()
    mlp.eval()
    
    # Text embedding
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        expected_hidden_states = num_layers_to_keep + 1
        if len(hidden_states) != expected_hidden_states:
            raise ValueError(f"Expected {expected_hidden_states} hidden states, but got {len(hidden_states)}. Check layer truncation.")
        
        hidden_states = hidden_states[-1]
    
    if use_first_token:
        cls_embeddings = hidden_states[:, 0]
    elif use_mean_pooling:
        cls_embeddings = torch.mean(hidden_states, dim=1)
    else:
        cls_embeddings = hidden_states[:, -1]
    
    logits = mlp(cls_embeddings)
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-7))
    
    predicted_class = np.argmax(probabilities)
    class_map = {0: "code-en", 1: "cot-en", 2: "dolly-en", 3: "math-en"}
    predicted_label = class_map[predicted_class]
    
    return predicted_label, probabilities, entropy

# An example
example_text = "Create an array to store integers from 1 to 10  var array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
predicted_label, probabilities, entropy = predict(example_text, model, tokenizer, mlp, device)
print(f"Predicted Label: {predicted_label}")
print(f"Probabilities: {probabilities}")
print(f"Entropy: {entropy}")