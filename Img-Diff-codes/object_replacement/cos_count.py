import json
import tqdm

new_json = []
for split_name in range(4):
    with open(f"./old_after_0625/filtered_file_new_caption_{str(split_name)}.txt", "r") as f:
        data = f.readlines()
        for temp_line in tqdm.tqdm(data):
            temp_json = eval(temp_line)
            # print(temp_json)
            if temp_json["cos"] > 0.9 and temp_json["cos"] <0.98:
                new_json.append(temp_json)

            # if temp_json["cos"] <= 0.9 or temp_json["cos"] >= 0.98:
            #     new_json.append(temp_json)
        
            # break

# print(new_json)
print(len(new_json))


# with open("filtered_file_new_edit_85_98.json", "w") as f: # filtered_file_new_edit_09_098.json
#     f.write(json.dumps(new_json))


data = new_json
new_json_0 = []
new_json_1 = []
new_json_2 = []
new_json_3 = []

length = len(data)
piece_len = int(length/4)

for idx, piece in tqdm.tqdm(enumerate(data)):
    if idx < piece_len:
        new_json_0.append(piece)
    elif idx >= piece_len and idx < 2*piece_len:
        new_json_1.append(piece)
    elif idx >= 2 * piece_len and idx < 3*piece_len:
        new_json_2.append(piece)
    else:
        new_json_3.append(piece)

# print(length)
print(piece_len)
print(len(new_json_0))
print(len(new_json_1))
print(len(new_json_2))
print(len(new_json_3))

with open("filtered_file_new_caption_9_98_0.json", "w") as f:
    f.write(json.dumps(new_json_0))

with open("filtered_file_new_caption_9_98_1.json", "w") as f:
    f.write(json.dumps(new_json_1))

with open("filtered_file_new_caption_9_98_2.json", "w") as f:
    f.write(json.dumps(new_json_2))

with open("filtered_file_new_caption_9_98_3.json", "w") as f:
    f.write(json.dumps(new_json_3))