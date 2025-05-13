import json
import os
import tqdm

if __name__ == "__main__":

    # docci
    new_data = []
    with open("./docci_descriptions.jsonlines", "r") as f:
        for temp_line in f:
            temp_json = json.loads(temp_line)
            new_data.append(temp_json)

    with open("./docci_descriptions.json", "w") as f:
        json.dump(new_data, f)


    # # flickr
    # new_data = []
    # with open("./flickr30k_train_localized_narratives.jsonl", "r") as f:
    #     for temp_line in f:
    #         temp_json = json.loads(temp_line)
    #         new_data.append(temp_json)

    # with open("./flickr_descriptions.json", "w") as f:
    #     json.dump(new_data, f)


    # # coco
    # new_data = []
    # with open("./coco_train_captions.jsonl", "r") as f:
    #     for temp_line in f:
    #         temp_json = json.loads(temp_line)
    #         new_data.append(temp_json)

    # with open("./coco_descriptions.json", "w") as f:
    #     json.dump(new_data, f)