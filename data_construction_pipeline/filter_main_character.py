import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_main_character_json', type=str, default="./input_main_character_json.json")
    parser.add_argument('--output_main_character_json', type=str, default="./output_main_character_json.json")

    args=parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    count_valid = 0
    new_json = []
    with open(args.input_main_character_json, "r") as f:
        data = json.load(f)
        for temp_line in data:
            try:
                if temp_line["count"] >= 4 and temp_line["count"] <= 8:
                    count_valid += 1
                    new_json.append(temp_line)
            except:
                print("broken")
                continue
    print(count_valid)

    with open(args.output_main_character_json, "a") as f:
        json.dump(new_json, f)
