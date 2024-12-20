import torch
from prompt_to_prompt_pipeline import Prompt2PromptPipeline
import tqdm
import argparse
import json
import os
from torch.utils.data import DataLoader, Dataset, SequentialSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="stable-diffusion-xl-base-1.0")
    parser.add_argument('--json_path', type=str, default="./data.json")
    parser.add_argument('--output_path', type=str, default="./output/")
    parser.add_argument("--local-rank", type=int)
    args=parser.parse_args()

    return args


class InferenceDataset(Dataset):

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    @torch.no_grad()
    def __getitem__(self, idx: int):
        return self.data[idx]["input"], self.data[idx]["output"], self.data[idx]["id"]


if __name__ == "__main__":
    args = parse_args()
    local_rank=args.local_rank
    # print(local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', init_method='env://')
    device = torch.device(f'cuda:{args.local_rank}') 

    model_path = args.model_path
    pipe = Prompt2PromptPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    seed = 864
    g_cpu = torch.Generator().manual_seed(seed)

    dataset = InferenceDataset(args.json_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=False)
    # pipe=torch.nn.parallel.DistributedDataParallel(pipe, device_ids=[args.local_rank])


    

    
    # file_name = args.json_path.replace("../gen_llava_", "").replace(".json", "")
    cross_attention_kwargs = {"edit_type": "refine",
                              "n_self_replace": 0.4,
                              "n_cross_replace": {"default_": 1.0, "confetti": 0.8},
                              }

    with torch.no_grad():
        for temp_idx, (temp_input, temp_output, temp_img_id) in enumerate(tqdm.tqdm(dataloader)):

            if "replaced" in temp_output[0].lower() or "modified" in temp_output[0].lower():
                continue
            
            if "---" in temp_output[0]:
                continue

            # try:
            prompts = [temp_input[0].strip("\""), temp_output[0].strip("\"")]
            image = pipe(prompts, cross_attention_kwargs=cross_attention_kwargs, generator=g_cpu)


            for idx, img in enumerate(image['images']):
                img.save(os.path.join(args.output_path, str(temp_img_id[0]) + f"_{str(idx)}.jpg"))
            # except:
            #     continue
