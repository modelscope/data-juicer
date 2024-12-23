# HumanVBench

## HumanVBench下载和测评方法
如需在HumanVBench上对模型进行测评，可以通过Evaluation.py实现。您可以修改eval_goal指定测评类型，可选包括：
'all'--对所有任务进行测评；你将获得每个问题的模型输出结果记录，每个任务的accuracy，每个任务的随机准确率，四类测评维度下分别的平均准确率。

One of 'Emotion Perception', 'Person Recognition', 'Human Behavior Analysis', 'Cross-Modal Speech-Visual Alignment'-- 对某一类维度进行测评；你将获得该类测评维度的模型输出结果记录，每个任务的accuracy，每个任务的随机准确率，该测评维度下分别的平均准确率。

'Self Selection'-- 可以自选需要测评的任务，任务内容请写在select_tasks中，可以从all_task_names中选择一个或多个。你将获得对于select_tasks的所有输出结果的记录，每个任务的accuracy，每个任务的随机准确率。


### 为了顺利完成测评，您需要补充的操作包括：
1. 在(benchmark存放网址)下载所有任务的jsonl文件，和每个任务的视频集合。将Evaluation.py中的task_jsonls_path指定为存放jsonl的存放路径。(task_jsonls_path下直接为17个jsonl), videos_save_path指定为存放所有出题视频的目录，存放格式满足

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

2. 完善模型加载部分（if __name__ == "__main__":中"load your model here"）,获得 model,processor,tokenizer

3. 完善模型推理部分（eval_on_one_task函数中"add your inference code", 获得输出文本outputs）

若调用API, 直接修改推理部分（删除关于model,processor,tokenizer的传递即可），能够获得回答文本输出outputs即可。

## 前置条件
参照datajuicer （https://github.com/modelscope/data-juicer） 的环境配置方法

## 数据Annotation Pipeline
将收集到原始视频存放于文件夹path/raw_videos/, 首先直接构建data_juicer能够读取的jsonl文件，每行内容如下：

{"videos":["path/raw_videos/video1.mp4"],"text":""}

然后就可以开始进行过滤和标注了，执行的yaml在/HumanVBenchRecipe/HumanCentricVideoAnnotationPipeline/process.yaml, 注意修改dataset_path为初始jsonl文件。

### 运行方式
python tools/process_data.py --config /HumanVBenchRecipe/HumanCentricVideoAnnotationPipeline/process.yaml


在上述标注结果的帮助下，可以构建HumanVBench中绝大多数任务。构建细节详见论文附录。我们会尽快公布17个任务的构建流程代码。

