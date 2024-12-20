import sys
sys.path.append("./FastChat")

import argparse
import json
import re
import time
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
import torch
import tqdm
from fastchat.model import get_conversation_template
import tqdm
import random
import transformers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vicuna_path', type=str, default="vicuna-13b-v1.5")
    parser.add_argument('--json_path', type=str, default="./data.json")
    parser.add_argument('--output_path', type=str, default="./output.json")
    args=parser.parse_args()

    return args

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    args = parse_args()
    answer_json = []

    model_path = args.vicuna_path
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16, _attn_implementation="flash_attention_2"
            ).half().cuda()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    for temp_caption in tqdm.tqdm(data):
        # temp_caption = temp_caption["conversations"][1]["value"]
        with torch.no_grad():
            with torch.inference_mode():
                # for temp_idx in range(5):

                msg = "Here is a sentence: \"" + temp_caption + "\". Please replace one entity in this sentence with another entity, such as an animal, a vehicle, or a piece of furniture. Please only answer with the replaced sentence."
                # print(msg)
                conv = get_conversation_template(model_path)
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], None)
                PROMPT = conv.get_prompt()
                ids = tokenizer.encode(PROMPT)
                input_ids = torch.LongTensor([ids]).to("cuda")


            
                seed_everything(random.randint(1,10000))
                
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8
                )
                out_text = tokenizer.decode(out[0])
                # out_text = tokenizer.batch_decode(out)
                    
                answer = out_text.replace(PROMPT, "").replace("\nEND", "").replace("</s>", "").replace("<s>", "").strip()


                if "replac" in answer.lower() or "modified" in answer.lower() or "become" in answer.lower():
                    continue

                if "---" in answer:
                    answer = answer.split("---")[-1].strip()


                # print(answer)
                # print(temp_caption)
                # print(answer)
                temp_json = {"input":temp_caption, "output":answer}

                # print(temp_json)
                answer_json.append(temp_json)

                temp_caption = answer

                # break

    with open(args.output_path, "w") as new_f:
        new_f.write(json.dumps(answer_json))