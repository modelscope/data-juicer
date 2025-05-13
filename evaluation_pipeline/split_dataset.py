import json
import os


if __name__ == "__main__":

    split_num = 4
    output_dir = "./DetailMaster_Dataset/"
    with open("./DetailMaster_Dataset/DetailMaster_Dataset.json", "r") as f:
        data = json.load(f)

    for i in range(split_num):

        with open(os.path.join(output_dir, f"split_final_prompt_{str(i)}.json"), "a") as f:
            if not i == split_num - 1:
                json.dump(data[i*int(len(data)/split_num):(i+1)*int(len(data)/split_num)], f)
            else:
                json.dump(data[i*int(len(data)/split_num):], f)