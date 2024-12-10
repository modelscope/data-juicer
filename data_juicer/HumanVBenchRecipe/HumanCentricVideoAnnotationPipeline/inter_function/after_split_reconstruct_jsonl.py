import json

def split_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            videos = record.get("videos", [])
            text = record.get("text", "")
            for video in videos:
                new_record = {"videos": [video], "text": text}
                outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")

input_jsonl = '/home/daoyuan_mm/data_juicer/HumanVBenchRecipe/dj_process_flow/raw_videos_prcocess1_j.jsonl'
output_jsonl = '/home/daoyuan_mm/data_juicer/HumanVBenchRecipe/dj_process_flow/raw_videos_prcocess1_j_reconstruct.jsonl'
split_jsonl(input_jsonl, output_jsonl)
