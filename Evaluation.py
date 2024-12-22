import os,json,tqdm,re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
# import things you need

# function for inference


def longest_common_substring(detect_text, total_text):
    m, n = len(detect_text), len(total_text)
    # Create a 2D table to store the lengths of common substrings
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    max_len = 0  # To store the length of the longest common substring
    end_index = 0  # To store the end index of the longest common substring in detect_text

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if detect_text[i - 1] == total_text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i

    # Extract the longest common substring
    longest_substring = detect_text[end_index - max_len:end_index]
    return longest_substring, max_len

def compare_output_answer(outputs, proper_option ,answer_dict):
    outputs = outputs.replace("Answer", "")
    translation_table = str.maketrans({",": "-", "-": ","})
    outputs_option = None
    if "best answer is Option " in outputs:
        outputs_option = outputs.split('best answer is Option ')[1][0]
    elif "best answer is option " in outputs:
        outputs_option = outputs.split('best answer is option ')[1][0]
    elif "best answer is " in outputs:
        print(1)

    if outputs_option == None:
        for key, value in answer_dict.items():
            if value in outputs:
                outputs_option = key
                break
            elif len(value) > 20:
                _, max_len = longest_common_substring(value, outputs)
                if abs(max_len-len(value)) <=2:
                    outputs_option = key
                    break
            else:
                new_value = value.translate(translation_table)
                if new_value in outputs:
                    outputs_option = key
                    break

    if outputs.split(' ')[0] == 'Based':
        outputs = outputs[5:]
    options = answer_dict.keys()
    if outputs_option == None:
        for char in outputs:
            if char in options:
                outputs_option = char
                break
    
    if outputs_option == proper_option:
        return True
    else:
        return False

def parse_options(input_text):
    lines = input_text.split("\n\n")
    options_dict = {}

    for line in lines:
        if 'Option' in line: 
            option_letter = line.split('Option ')[-1][0]
            content = re.findall(r"\{(.*?)\}", line)[0]
            options_dict[option_letter] = content
    
    return options_dict

def eval_on_one_task(video_path_root, task_jsonls_path, task_name, model,processor,tokenizer,round_num):
    with open(os.path.join(task_jsonls_path, f"{task_name}.jsonl"), "r") as f:
        lines = f.readlines() 

    result_dict = []
    answer_count = 0
    correct_count = 0
    option_number = 0
    for temp_piece in tqdm.tqdm(lines, desc=f"{task_name}: Round {round_num}", unit="question"):
        answer_count += 1
        line = temp_piece.strip()
        line = json.loads(line)
        
        prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter of the correct option.\n\n\n\n" + line["Question"] + " " + line["choices"] + "\n\n Only answer best answer's option letter. Your option is: "
        video_path = os.path.join(video_path_root, line["Video"])

        '''
        add your inference code

        outputs = single_test(model,
                            processor,
                            tokenizer,
                            video_path,
                            qs=prompt,
                            pre_query_prompt=pre_query_prompt,
                            num_frames=num_frames,
                            conv_mode=conv_mode).replace("Answer", "")
        print(outputs)
        '''
        
        print(outputs)
        option_dict = parse_options(line["choices"])
        # print(line["choices"])
        option_number += len(line["choices"].split('\n\n'))-1

        result_json = {}
        result_json["id"] = line["Question_id"]
        result_json["response"] = outputs
        result_json["proper_option"] = line["Answer"]
        result_json["proper_option_content"] = option_dict[line["Answer"]]


        if compare_output_answer(outputs, line["Answer"], option_dict):
            correct_count += 1
            result_json["correct"] = True
        else:
            result_json["correct"] = False

        result_dict.append(result_json)
    
    return result_dict, correct_count, answer_count, option_number

def HumanVBench_eval(task_jsonls_path, 
                     video_path_root,
                     fold_time,
                     eval_goal,
                     select_tasks,
                     result_save_path,
                     model,processor,tokenizer):
    
    acc_dict = {}
    random_acc_dict = {}
    for fold_num in range(fold_time):
        all_tasks_result = []

        for name in select_tasks:
            print(f'Evaluation on task: {name}, fold: {fold_num}')
            if name not in acc_dict.keys():
                acc_dict[name] = []
            result_json, correct_count, answer_count, option_number = eval_on_one_task(video_path_root, task_jsonls_path, name, model,processor,tokenizer, fold_num)
            all_tasks_result = all_tasks_result + result_json
            acc = correct_count / answer_count
            if name not in random_acc_dict.keys():
                random_acc_dict[name] = answer_count / option_number
            acc_dict[name].append(acc)


        with open(os.path.join(result_save_path, f'fold{str(fold_num)}_response.json'), 'a') as file:
            json.dump(all_tasks_result, file, indent=1)

    ave = {}
    for key in list(acc_dict):
        ave[key] = np.array(acc_dict[key]).mean()

    if not eval_goal == 'all':
        with open(os.path.join(result_save_path, 'avg_modelname_result.json'), 'w') as file:
            json.dump(acc_dict, file, indent=1)
            json.dump(ave, file, indent=1)
            json.dump(random_acc_dict, file, indent=1)
        print(f'Evaluation Finish!')

    elif eval_goal == 'all':
        ave_2 = {}
        ave_2['Emotion Perception'] = (ave["Emotion_Recognition"] + ave["Emotion_Temporal_Analysis"] + ave["Attitude_Recognition"] + ave["Emotion_Intensity_Compare"] + ave["Emotion_Recognition_in_Conversation"])/5
        ave_2['Person Recognition'] = (ave["Text2Human"] + ave["Human2Text"] + ave["Human_Count"] + ave["Appearance_Time_Detection"])/4
        ave_2['Human Behavior Analysis'] = (ave["Behavior_Temporal_Analysis"] + ave["Behavior_Causualty_Analysis"] + ave["Action_at_Specified_Time"] + ave["Time_of_Specific_Action"])/4
        ave_2['Cross-Modal Speech-Visual Alignment'] = (ave["Audio_Visual_Speaker_Matching"] + ave["Active_Speaker_Detection"] + ave["Audio_Visual_Alignment_Detection"] + ave["Speech_Content_Matching"])/4
        with open(os.path.join(result_save_path, 'avg_modelname_result.json'), 'w') as file:
            json.dump(acc_dict, file, indent=1)
            json.dump(ave, file, indent=1)
            json.dump(random_acc_dict, file, indent=1)
            json.dump(ave_2, file, indent=1)
        print(f'Evaluation Finish!')
    
    elif eval_goal == 'Emotion Perception':
        ave_2 = {}
        ave_2['Emotion Perception'] = (ave["Emotion_Recognition"] + ave["Emotion_Temporal_Analysis"] + ave["Attitude_Recognition"] + ave["Emotion_Intensity_Compare"] + ave["Emotion_Recognition_in_Conversation"])/5
        with open(os.path.join(result_save_path, 'avg_modelname_result.json'), 'w') as file:
            json.dump(acc_dict, file, indent=1)
            json.dump(ave, file, indent=1)
            json.dump(random_acc_dict, file, indent=1)
            json.dump(ave_2, file, indent=1)
        print(f'Evaluation Finish!')

    elif eval_goal == 'Person Recognition':
        ave_2 = {}
        ave_2['Person Recognition'] = (ave["Text2Human"] + ave["Human2Text"] + ave["Human_Count"] + ave["Appearance_Time_Detection"])/4
        with open(os.path.join(result_save_path, 'avg_modelname_result.json'), 'w') as file:
            json.dump(acc_dict, file, indent=1)
            json.dump(ave, file, indent=1)
            json.dump(random_acc_dict, file, indent=1)
            json.dump(ave_2, file, indent=1)
        print(f'Evaluation Finish!')

    elif eval_goal == 'Human Behavior Analysis':
        ave_2 = {}
        ave_2['Human Behavior Analysis'] = (ave["Behavior_Temporal_Analysis"] + ave["Behavior_Causualty_Analysis"] + ave["Action_at_Specified_Time"] + ave["Time_of_Specific_Action"])/4
        with open(os.path.join(result_save_path, 'avg_modelname_result.json'), 'w') as file:
            json.dump(acc_dict, file, indent=1)
            json.dump(ave, file, indent=1)
            json.dump(random_acc_dict, file, indent=1)
            json.dump(ave_2, file, indent=1)
        print(f'Evaluation Finish!')
    elif eval_goal == 'Cross-Modal Speech-Visual Alignment':
        ave_2 = {}
        ave_2['Cross-Modal Speech-Visual Alignment'] = (ave["Audio_Visual_Speaker_Matching"] + ave["Active_Speaker_Detection"] + ave["Audio_Visual_Alignment_Detection"] + ave["Speech_Content_Matching"])/4
        with open(os.path.join(result_save_path, 'avg_modelname_result.json'), 'w') as file:
            json.dump(acc_dict, file, indent=1)
            json.dump(ave, file, indent=1)
            json.dump(random_acc_dict, file, indent=1)
            json.dump(ave_2, file, indent=1)
        print(f'Evaluation Finish!')



Evaluation_items = ['all', 'Emotion Perception', 'Person Recognition', 'Human Behavior Analysis', 'Cross-Modal Speech-Visual Alignment', 'Self Selection']
all_task_names = ["Emotion_Recognition", "Emotion_Temporal_Analysis", "Attitude_Recognition", "Emotion_Intensity_Compare", "Emotion_Recognition_in_Conversation",
                  "Text2Human", "Human2Text", "Human_Count", "Appearance_Time_Detection",
                  "Behavior_Temporal_Analysis", "Behavior_Causualty_Analysis", "Action_at_Specified_Time", "Time_of_Specific_Action",
                  "Audio_Visual_Speaker_Matching", "Active_Speaker_Detection", "Audio_Visual_Alignment_Detection", "Speech_Content_Matching"]

if __name__ == "__main__":
    '''
    load your model here
    '''
    # The following interfaces are used to control the evaluation
    task_jsonls_path = None  # './task_jsonls'
    videos_save_path = None  # './HumanVBench_videos'
    eval_goal = None  # select one from Evaluation_items
    select_tasks = []  # when eval_goal == 'Self Selection' fill it. 
    result_save_path = None 
    fold_time = 5  # random evaluation times
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)


    if eval_goal == 'Emotion Perception':
        select_tasks = ["Emotion_Recognition", "Emotion_Temporal_Analysis", "Attitude_Recognition", "Emotion_Intensity_Compare", "Emotion_Recognition_in_Conversation"]
    if eval_goal == 'Person Recognition':
        select_tasks = ["Text2Human", "Human2Text", "Human_Count", "Appearance_Time_Detection"]
    if eval_goal == 'Human Behavior Analysis':
        select_tasks = ["Behavior_Temporal_Analysis", "Behavior_Causualty_Analysis", "Action_at_Specified_Time", "Time_of_Specific_Action"]
    if eval_goal == 'Cross-Modal Speech-Visual Alignment':
        select_tasks = ["Audio_Visual_Speaker_Matching", "Active_Speaker_Detection", "Audio_Visual_Alignment_Detection", "Speech_Content_Matching"]
    if eval_goal == 'all':
        select_tasks = all_task_names

    HumanVBench_eval(task_jsonls_path, videos_save_path, fold_time, eval_goal, select_tasks, result_save_path, model,processor,tokenizer)


