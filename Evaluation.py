import os,json,tqdm
import numpy as np
# import things you need

# function for inference

def eval_on_one_task(video_path_root, task_jsonls_path, task_name, model,processor,tokenizer):
    with open(os.path.join(task_jsonls_path, f"{task_name}.jsonl"), "r") as f:
        result_dict = []
        answer_count = 0
        correct_count = 0
        option_number = 0
        for temp_piece in tqdm.tqdm(f):
            answer_count += 1
            line = temp_piece.strip()
            line = json.loads(line)
            
            prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter of the correct option.\n\n\n\n" + line["Question"] + " " + line["choices"] + "\n\n The best answer is:"
            video_path = os.path.join(video_path_root, line["Video"])

            '''
            # add your inference code

            outputs = single_test(model,
                                processor,
                                tokenizer,
                                video_path,
                                qs=prompt,
                                pre_query_prompt=pre_query_prompt,
                                num_frames=num_frames,
                                conv_mode=conv_mode).replace("Answer", "")
            # get outputs
            '''
            
            option_number += len(line["choices"].split('\n\n'))-1

            result_json = {}
            result_json["id"] = line["Question_id"]
            result_json["response"] = outputs
            result_json["real_answer"] = line["Answer"]


            if line["Answer"] in outputs:
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
            result_json, correct_count, answer_count, option_number = eval_on_one_task(video_path_root, task_jsonls_path, name, model,processor,tokenizer)
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
        ave_2['Emotion Perception'] # = 5个平均
        ave_2['Person Recognition']
        ave_2['Human Behavior Analysis']
        ave_2['Cross-Modal Speech-Visual Alignment']
        



Evaluation_items = ['all', 'Emotion Perception', 'Person Recognition', 'Human Behavior Analysis', 'Cross-Modal Speech-Visual Alignment', 'Self Selection']
all_task_names = ["Time_of_Specific_Action", "Appearance_time_recognition", "Audio_Visual_Alignment_Moment_Detection", "MMERC_ready_for_eval", "MMERC_ready_for_eval_only_target",
                        "Speaker_Audio_matching_156", "Speech_content_matching", "Text2Human", "Action_at_specific_time", "Action_Temporal_Analysis",
                        "Active_speaker_detection", "Attitude_Recognition", "Behavior_Causualty_Analysis", "Emotion_Intensity_Compare", "Emotion_Temporal_Analysis",
                        "Human_Count_raw", "Human_Emotion_Recognition", "Human_Emotion_Recognition_speaker", "Human2Text"]

if __name__ == "__main__":
    '''
    load your model here
    '''
    # The following interfaces are used to control the evaluation
    task_jsonls_path = None  # '/home/daoyuan_mm/HumanCentricBenchmark/AAReallyFinalBenchmark/for_Eval'
    videos_save_path = None  # '/mnt/daoyuan_open_research/zt_data/zt_videos/HumanCentricVideoBenchmark'
    eval_goal = None  # select one from Evaluation_items
    select_tasks = []  # when eval_goal == 'Self Selection' fill it. 
    result_save_path = None  # '/home/daoyuan_mm/tempt3'
    fold_time = 2  # random evaluation times

    
    if eval_goal == 'Emotion Perception':
        select_tasks = ["A", "B", "C"]
    if eval_goal == 'Person Recognition':
        select_tasks = ["A", "B", "C"]
    if eval_goal == 'Human Behavior Analysis':
        select_tasks = ["A", "B", "C"]
    if eval_goal == 'Cross-Modal Speech-Visual Alignment':
        select_tasks = ["A", "B", "C"]
    if eval_goal == 'all':
        select_tasks = all_task_names

    HumanVBench_eval(task_jsonls_path, videos_save_path, fold_time, eval_goal, select_tasks, result_save_path, model,processor,tokenizer)


