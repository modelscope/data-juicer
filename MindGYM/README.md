# MindGYM

Implementation of the paper _**MindGYM: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions**_.

> Large vision-language models (VLMs) face challenges in achieving robust, transferable reasoning abilities due to reliance on labor-intensive manual instruction datasets or computationally expensive self-supervised methods. To address these issues, we introduce MindGYM, a framework that enhances VLMs through synthetic self-challenging questions, consisting of three stages: (1) Seed Single-Hop Question Synthesis, generating cognitive questions across textual (e.g., logical deduction) and multimodal contexts (e.g., diagram-based queries) spanning eight semantic areas like ethical analysis; (2) Challenging Multi-Hop Question Synthesis, combining seed questions via diverse principles like bridging, visual-textual alignment, to create multi-step problems demanding deeper reasoning; and (3) Thinking-Induced Curriculum Fine-Tuning, a structured pipeline that progressively trains the model from scaffolded reasoning to standalone inference. By leveraging the model's self-synthesis capability, MindGYM achieves high data efficiency (e.g., +16% gains on MathVision-Mini with only 400 samples), computational efficiency (reducing both training and inference costs), and robust generalization across tasks. Extensive evaluations on seven benchmarks demonstrate superior performance over strong baselines, with notable improvements (+15.77% win rates) in reasoning depth and breadth validated via GPT-based scoring. MindGYM underscores the viability of self-challenging for refining VLM capabilities while minimizing human intervention and resource demands. Code and data are released to advance multimodal reasoning research.

![overview](https://github.com/user-attachments/assets/2bd539d8-5afe-4199-b26a-ae376f48d0b4)

_**The width- and breadth-based scoring and self-challenging synthesis OPs are being incorporated into Data-Juicer main branch.**_

The models and datasets used in the paper are as follows: 

From Huggingface: 
[Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)
[OK-VQA](https://huggingface.co/datasets/lmms-lab/OK-VQA)

From Modelscope:
[Qwen2.5-VL-7B](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)

## Step 0. Installation

### Conda Environment

```bash
conda create -n mindgym python=3.10
```

### Pip Installation

```bash
pip install -r requirements.txt
```

## Step 1. Self-challenging QAs Generation

Data synthesis is divided into text and image ends, both of which are implemented in the form of pipline, and only one call is needed to synthesize data. The file is located in `./data_generate`, which can realize data synthesis in Chinese, English, text, and images. We integrated _Seed Single-Hop Question Synthesis_ and _Challenging Multi-Hop Question Synthesis_ into one pipeline.

For text data:
```bash
bash data_generate.sh
bash data_generate_cn.sh
```

For image data (taking ScienceQA as an example):
```bash
bash data_generate_img.sh
bash data_generate_img_cn.sh
```

## Step 2. TrainFormatPrepare

After generating data using the code in `data_generate`, the data needs to be preprocessed. The file is located in `./data_process` to adapt to the training data format of [ms-swift](https://github.com/modelscope/ms-swift).

For text data:
```python
python ./data_process/process.py --input_en_path \[input_file_en\] --input_cn_path [input_file_cn]
```

For image data (note that the path of the image needs to be modified accordingly):
```python
python ./data_process/process_img.py --input_en_path \[input_file_en\] --input_cn_path [input_file_cn]
```

The following eight training data for course learning will be obtained (taking text data as an example):

- _Question + Thinking -> Answer_
- _Question + Answer -> Thinking_
- _Question + Answer -> Thinking_
- _Question -> Answer_

![image](https://github.com/user-attachments/assets/a2967eae-d783-4db3-b494-3400526aeacd)

## Step 3. Thinking-Induced Curriculum Fine-Tuning

Curriculum Learning: Training is divided into four steps

_Question + Thinking -> Answer_: Corresponding to the above `cn_with_0306_0.json`

```bash
bash ./train/step-1.sh
```

_Question + Answer -> Thinking_: Corresponding to the above `cn_with_0306_1.json`

```bash
bash ./train/step-2.sh
```

_Question + Answer -> Thinking_: Corresponding to the above `cn_with_0306_2.json`

```bash
bash ./train/step-3.sh
```

_Question -> Answer_: Corresponding to the above `cn_with_0306_3.json`

```bash
bash ./train/step-4.sh
```

For image data, you need to set the `--freeze_aligner` parameter in the `bash` file to `False` to align the text model and the visual model.

## Step 4. The Implementation of Eval Module

For the detailed implementation of evaluation, please refer to `./eval`.

### Installation

To ensure the latest capabilities of ms-swift are used, it is recommended to install from the GitHub repository.

1. **Conda Environment:**

```bash
conda create -n ms-eval python=3.10
```

2. **Install from source:**

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e '.[eval]'
```

3. **If replication is needed, the core dependencies are:**

```bash
python=3.10, evalscope==0.11.0, ms-swift==3.2.0, ms-vlmeval==0.0.13, torch==2.5.1, transformers==4.49.0
```

For detailed `pip list`, please refer to `./eval/requirements.txt`. Note that an additional library that may need to be installed is `qwen-vl-utils` for Qwen2.5-VL, and `timm` for Intern2.5-VL.

### Evaluation

We have listed the detailed evaluation scripts in this folder, including the Qwen2.5-VL series and Intern2.5-VL series. As an example, we focus on the evaluation script for Qwen2.5-VL-7B-Instruct located in the `./eval/qw25/` folder.

1. **Multimodal Evaluation**

We selected four datasets: `MMStar`, `MathVision`, and `MathVista`, to conduct comprehensive evaluations. The `--eval_backend`is set to `VLMEvalKit`. We conduct full evaluations on the other datasets (`./eval/qw25/mm-eval-1.sh`), resulting in two separate scripts. Ensure that the `OPENAI_API_KEY` is set correctly.

2. **Text-based Evaluation**

We selected `GSM8K`, `MATH`, and `GPQA` for comprehensive evaluation. The `--eval_backend` is set to `Native`. Similarly, except for `GPQA` (`./eval/qw25/text-eval-2.sh`), where we set a limit of 300 due to time constraints, we conduct full evaluations on the other datasets (`./eval/qw25/text-eval-1.sh`), resulting in two separate scripts.

(To avoid repetitive operations, can we ignore the time constraints and perform a full evaluation? It would take an additional 2-3 hours approximately.)

Two points to note:

- Keep `eval_num_proc` set to `1`. Increasing the value will only increase the number of ports used without speeding up the process, and too many ports may cause port conflicts and self-locking.
- The `--port` setting: Ports are theoretically allocated automatically, but if you need to interrupt in the middle, uncleared ports might cause conflicts. Therefore, it is recommended to set the ports manually.

## Reference

If you find our work useful for your research or development, please kindly cite the following [paper](https://arxiv.org/abs/2503.09499).

```bib
@misc{xu2025mindgymenhancingvisionlanguagemodels,
      title={MindGYM: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions}, 
      author={Zhe Xu and Daoyuan Chen and Zhenqing Ling and Yaliang Li and Ying Shen},
      year={2025},
      eprint={2503.09499},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.09499}, 
}
```
