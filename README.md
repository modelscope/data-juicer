# HumanVBench

## Prerequisites
Follow the environment setup instructions from datajuicer (https://github.com/modelscope/data-juicer).  

## Data Annotation Pipeline
Store the collected raw videos in the folder `path/raw_videos/`. Create a JSONL file that can be read by data_juicer. Each line should have the following format:  

{"videos":["path/raw_videos/video1.mp4"],"text":""}

Then you can start filtering and annotating process. The executed yaml is in /HumanVBenchRecipe/HumanCentricVideoAnnotationPipeline/process.yaml. Note that dataset_path should be changed to the above generated initial jsonl file.

### Execution
python tools/process_data.py --config /HumanVBenchRecipe/HumanCentricVideoAnnotationPipeline/process.yaml


With the help of the annotated results, most tasks in HumanVBench can be constructed. Details about the construction process are provided in the paper's appendix. We will release the code for constructing all 17 tasks as soon as possible.


## HumanVBench Download and Evaluation
To evaluate a model on HumanVBench, use the Evaluation.py script. You can modify the eval_goal parameter to specify the evaluation type. Options include:

'all': Evaluate on all tasks. You will receive: model output results for each question; accuracy for each task; random accuracy for each task; average accuracy across four evaluation dimensions.

One of 'Emotion Perception', 'Person Recognition', 'Human Behavior Analysis', 'Cross-Modal Speech-Visual Alignment'-- Evaluate on a specific dimension. You will receive: model output results for tasks within the selected dimension; accuracy for each task; random accuracy for each task; average accuracy of the selected dimension.

'Self Selection'-- Evaluate on selected tasks. You should specify the tasks you want for the evaluation, choosing one or more from all_task_names. You will receive: model output results for the selected tasks; accuracy for each task; random accuracy for each task.


### Steps to Complete the Evaluation
1. Download all task JSONL files and the corresponding video collections from the benchmark download link. Set task_jsonls_path in Evaluation.py to the directory where the JSONL files are stored. The directory should directly contain 17 JSONL files. Set videos_save_path to the directory where all the task-related videos are stored. The directory structure should meet the following format:

(videos_save_path)--  
&nbsp;&nbsp;&nbsp;&nbsp;Action_Temporal_Analysis--  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（.mp4）  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（.mp4）  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（.mp4）  
&nbsp;&nbsp;&nbsp;&nbsp;Active_speaker_detection--  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（.mp4）  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（.mp4）  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（.mp4）  
&nbsp;&nbsp;&nbsp;&nbsp; ...  

2. Complete the model loading section (in the if __name__ == "__main__": block under "load your model here") to obtain model, processor, and tokenizer.

3. Implement the model inference logic (in the eval_on_one_task function under "add your inference code") to generate the output text (outputs).

If using an API, directly modify the inference section (removing references to model, processor, and tokenizer). Ensure the API can produce the output text (outputs).
