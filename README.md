# DetailMaster: Can Your Text-to-Image Model Handle Lengthy Prompts?

## **README under construction!!!**

> **Abstract:** While recent text-to-image (T2I) models show impressive capabilities in synthesizing images from brief descriptions, their performance significantly degrades when confronted with lengthy, detail-intensive prompts required in professional applications. We present DetailMaster, the first comprehensive benchmark specifically designed to evaluate T2I systems' abilities to handle extended textual inputs that contain complex compositional requirements. Our framework introduces four critical evaluation dimensions: Character Attributes, Structured Character Locations, Multi-Dimensional Scene Attributes, and Explicit Spatial/Interactive Relationships. The benchmark comprises lengthy and detail-rich prompts averaging 284.89 tokens, with high quality validated by expert annotators. We evaluate 7 general-purpose and 5 lengthy-prompt-optimized T2I models, revealing critical performance limitations: state-of-the-art systems achieve merely 50% accuracy in key dimensions like attribute binding and spatial reasoning, while all models showing progressive performance degradation as prompt length increases. Our analysis highlights systemic failures in structural comprehension and detail overload handling, motivating future research into architectures with enhanced compositional reasoning. We open-source our dataset, data construction code, and evaluation tools to advance the field of lengthy-prompt-driven and detail-rich T2I generation.



## Environment

```bash
$ pip install -r requirements.txt
```





## Evaluation

### Step1. Generate your images

Please generate your images based on our [prompts](https://github.com/modelscope/data-juicer/tree/DetailMaster/DetailMaster_Dataset). (The key name for our prompts is "polished_prompt" in the DetailMaster dataset)

During the image generation, you need to simultaneously **record metadata for each generated image**. Every generated image must include two key attributes: 

1. ***"output_image_name":*** The filename of your generated image;
2. ***"image_id":*** A concatenation of DetailMaster data samples' "dataset_target" and "image_id". Formally, for each sample represented as `temp` in the DetailMaster dataset, the ***"image_id"*** should be constructed as `"temp['dataset_target']_temp['image_id']"`. 

You should compile the results into a JSON file with the following structure:

```
[{"output_image_name": "xxx1.jpg", "image_id": "docci_qual_dev_00003.jpg"}, 
 {"output_image_name": "xxx2.jpg", "image_id": "docci_qual_dev_00008.jpg"},
 ...
 {"output_image_name": "xxx500.jpg", "image_id": "ln_flickr_294391131.jpg"},
 ...
 {"output_image_name": "xxx4116.jpg", "image_id": "ln_coco_000000298347.jpg"}]
```

​	**For reference, we provide [sample generation code](https://github.com/modelscope/data-juicer/tree/DetailMaster/evaluation_pipeline/image_generation_example) based on Stable Diffusion 1.5.**



### Step2. Split the dataset (Optional)

Our evaluation process requires 20GB–39GB of GPU memory (varies with `cache_max_entry_count` parameter values between 0.01–0.8). On a single NVIDIA L20 GPU, the evaluation process takes approximately 10 hours when `cache_max_entry_count=0.8`.

To reduce evaluation time, we provide dataset partitioning code. Please run:

```shell
$ python evaluation_pipeline/split_dataset.py
```



### Step3. Run the evaluation code

To run the evaluation code, you need to provide ***the directory path containing your model's generated images*** and ***the generated image metadata file***. In addition, you should assign your model name to the `output_name_prefix`.

If you have performed dataset splitting using our "split dataset" code, you should specify the split subsets in the `ann_json_path` and ensure the `output_log_dir` points to the same output folder.

Please run:

```shell
$ bash evaluation_pipeline/eval_process.sh
```



### Step4. Collect the evaluation results

To collect the evaluation results, you should set the `eval_output_log_dir_name` to the path of your evaluation output folder, and assign your model name to the `name_prefix`.

Please run:

```shell
$ bash evaluation_pipeline/cal_eval.sh
```

Now, you get the evaluation results.





## Dataset Construction  

We provide the codes of our dataset construction pipeline. To reproduce our dataset, please run the following scripts in sequential order.

### Step1: Preparation

To get started, you first need to download the annotation files and images for [DOCCI](https://google.github.io/docci/) and [Localized Narratives](https://google.github.io/localized-narratives/) (Flickr30k and COCO subsets). Regarding the annotation files, they are originally in jsonl format and need to be converted to json format. 

Please run:

```shell
$ python playground/process_jsonlines.py
```



### Step2: Main character identification

"Main character identification" refers to extracting a list of main characters for each sample based on the source image and the original caption.

Please run:

```shell
$ bash /data_construction_pipeline/detect_main_character.sh
```



### Step3: Main character filtering

"Main character filtering" refers to sample filtering based on the number of main characters present.

Please run:

```shell
$ bash data_construction_pipeline/filter_main_character.sh
```



### Step4: Character localization

"Character localization" refers to detecting the positions of main characters and extracting their bounding box information by jointly leveraging an MLLM and the open-set detector YOLOE.

Please run:

```shell
$ bash data_construction_pipeline/detect_character_locations.sh
```



### Step5: Spatial partitioning  

"Spatial partitioning" refers to converting the bounding box information of main characters into positional descriptions.

Please run:

```shell
$ bash data_construction_pipeline/bbox_to_location.sh
```



### Step6: Character attribute extraction

"Character attribute extraction" refers to extracting character attributes based on the original caption and the cropped image of the character.

Please run:

```shell
$ bash data_construction_pipeline/detect_character_attributes.sh
```



### Step7: Scene attribute and entity relationship detection

"Scene attribute and entity relationship detection" refers to using an MLLM to extract background, lighting, and style conditions from source images, while also providing the MLLM all main character names to analyze their spatial or interactive relationships.

Please run:

```shell
$ bash data_construction_pipeline/detect_scene_attributes.sh
```



### Step8: Polish prompt

"Polish prompt" refers to incorporating all extracted attributes into the original captions to generate detailed, lengthy prompts.

Please run:

```shell
$ bash data_construction_pipeline/polish_prompt.sh
```



### Step9: Filter Features

"Filter features" refers to analyzing whether the extracted features have been successfully incorporated into the polished prompts, and removing any features that failed to be integrated.

Please run:

```shell
$ bash data_construction_pipeline/filter_feature.sh
```



### Step10: Get valid prompts

"Get valid prompts" refers to filtering and obtaining valid prompts based on the number of `character_attributes` and `character_locations` for each evaluation sample.

Please run:

```shell
$ python data_construction_pipeline/final_filter_valid_prompt.py
```

