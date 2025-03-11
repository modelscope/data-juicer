# DaaR

Implementation of the paper ***Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data.***

> Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains.
In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance.
To address these challenges, in this paper, we study the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations for both inter- and intra-diversity. 
Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data.
Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-development for LLMs.


---

## Installation

The training platform relies on [ms-swift](https://github.com/modelscope/ms-swift/tree/release/2.5), and the evaluation platform relies on [OpenCompass](https://github.com/open-compass/OpenCompass/). Both can be directly installed via pip. The detailed steps are as follows:

**Conda Environment:**

```bash
conda create -n daar python=3.10
```

**Pip Installation**

To consistently reproduce the results of this paper on both training and evaluation sides, the core environment dependencies are as follows: `CUDA==12.4`, `python==3.10`, `ms-swift==2.5.0.post1`, `opencompass==0.3.3`, `torch==2.4.1`, `transformers==4.45.2`. Please note that these dependencies pertain **only to the training and evaluation environments**. The construction of the data pool and the implementation of the DaaR method **do not strictly depend** on this environment. We have bundled the environment into a `./requirements.txt` file, which can be directly installed via pip:

```bash
pip install -r requirements.txt
```

## Construction of Data Pool

As mentioned in the paper, we selected four basic capability datasets: [Dolly-15K](https://modelscope.cn/datasets/AI-ModelScope/databricks-dolly-15k) for *Common Sense*, [Cot-en](https://modelscope.cn/datasets/YorickHe/CoT) for *Reasoning*, [Math-Instruct](https://modelscope.cn/datasets/AI-ModelScope/MathInstruct) for *Mathematics* and [Code-Alpaca](https://modelscope.cn/datasets/AI-ModelScope/CodeAlpaca-20k) for *Coding* to construct the data pool.

Each dataset was initially filtered and randomly reduced to 10,000 entries, resulting in a combined data pool of 40,000 entries. Specifically, for the [Math-Instruct ]([Math-Instruct](https://modelscope.cn/datasets/AI-ModelScope/MathInstruct))dataset, due to its inclusion of CoT and certain coding capabilities, we extract a highly mathematics-related subset and use regular expressions to filter out the coding-related content (including `program`, `python`, `def`, `import`, `print`, `return`), ensuring it remains within the domain of mathematics.

As a result, we have provided the original data pool instruction training datasets for the four domains in `./data/raw`. All subsequent experiments are based on this data pool. And we will utilize `Qwen2.5-7B` as an example. For each piece of data in the data pool, it first needs to be vectorized into corresponding embeddings using the **LLM's Embedding Layer** for all subsequent processing. The code for this, located in `./distribution_syn/ebd_save.py`, is as follows:

```bash
python ./distribution/ebd_save.py --model_path ./models/Qwen2.5-7B --output_path ./data/ebd/qw25
```

## Contrastive Distribution Synthesis

### Inter-Diversity

For constructing datasets with different distributions through **inter-diversity** as mentioned in the paper, we have integrated this functionality into `./distribution_syn/select_inter_diversity.py`. The command to construct data distributions similar to those shown in **Fig. 9 (c) and (d)** is as follows:

```bash
python distribution_syn/select_inter_diversity.py --input_path ./data/ebd/qw25 --lower 0 --upper 2000 --output_path ./data/res/qw25/inter_diversed

python distribution_syn/select_inter_diversity.py --input_path ./data/ebd/qw25 --lower 8000 --upper 10000 --output_path ./data/res/qw25/inter_closed
```

Note that `--input_path` should be set to the location where the embeddings are stored, and `--lower` and `--upper` control the granularity of the selected data distribution. The results in **Table 8** of the paper were achieved by controlling intervals of 2000.

### Intra-Diversity

For constructing datasets with different distributions through **intra-diversity** as mentioned in the paper, we have integrated this functionality into `./distribution_syn/select_intra_diversity.py`. The command to construct data distributions similar to those shown in **Fig. 9 (e) and (f)** is as follows:

```bash
python distribution_syn/select_intra_diversity.py --input_path ./data/ebd/qw25 --lower 0 --upper 2000 --output_path ./data/res/qw25/intra_diversed

python distribution_syn/select_intra_diversity.py --input_path ./data/ebd/qw25 --lower 8000 --upper 10000 --output_path ./data/res/qw25/intra_closed
```

Note that `--input_path` should be set to the location where the embeddings are stored, and `--lower` and `--upper` control the granularity of the selected data distribution. The results in **Table 8** of the paper were achieved by controlling intervals of 2000.

## Implementation of DaaR

### Step 1: Centroid Construction

(Instruct explaination) **First**, we need to obtain an initial seed dataset for each domain (5 entries per domain). This will generate `txt` and `jsonl` folders at the same level as the executable file under `./warmup_seed/qw25`. Although we use `jsonl` for subsequent operations, due to difficulties in extraction using regular expressions, we recommend using a **prompt-based approach** to extract data into `jsonl` format (we are currently refining this method). To ensure the model generates outputs correctly, we use the Instruct version of the same-spec LLM for seed generation. A reference prompt is: *'I am working on a conversion from txt to jsonl file. The following is a txt file containing five instruction pairs. Please convert it into the following jsonl format and return code block directly.
{"instruction": "", "input": "", "output": ""}'*

```bash
python ./daar/1_centroid/1_warmup.py --model_path ./models/Qwen2.5-7B-Instruct
```

**Next**, we proceed to the formal generation of **model-aware** domain data. For each domain, adjust the arguments sequentially and execute the following command-line code. Note that **empirical validation** with only 10 samples is sufficient to represent the semantic information of the domain. To accelerate generation, you can slightly increase the `--similarity_threshold` parameter. The generated files will be saved in the same directory under `./syn/qw25`.

```bash
python ./daar/1_centroid/2_syn.py \\
    --model_path ./models/Qwen2.5-7B-Instruct \\
    --selected_domain coding \\
    --save_dir ./daar/1_centroid/syn/qw25/coding.jsonl \\
    --warmup_file ./daar/1_centroid/warmup_seed/qw25/jsonl/coding_warmup.jsonl \\
    --gen_num 10 \\
    --similarity_threshold 0.85
```

**Next**, we need to extract semantic information from the synthesized data of four types of domains via `embedding`:

```bash
python ./daar/1_centroid/3_syn_ebd.py --model_path ./models/Qwen2.5-7B
```

**Finally**, we obtain the DaaR Probe training data with model-aware labels. Before executing this step, you need to sample your actual LLM SFT data to obtain a dataset with the same distribution. Then, use the embedding method mentioned above to generate embedding information. Place both of these files in the following directory: `./daar/1_centroid/train_data/[models]/[raw_data.jsonl, ebd.npz]`. In our study, we selected 5,000 data points that share the same distribution as the SFT data but remain isolated, ensuring both rigor and generalization.

```bash
python ./daar/1_centroid/4_kmeans.py
```

### Step 2: Probe Training

***1-Domain Discrimination:*** We use the labeled `train_data.jsonl` obtained in the previous step to train the **discrimination-probe**. The training is conducted using cross-entropy loss. The execution command is as follows (refer to the argparse settings in the script for specific parameters). After training, we perform inference on the probe-training data to obtain the softmax scores for each model-aware domain.

```bash
# Training discrimination probe
python ./daar/2_train/1_ce_train.py

# Infering
python ./daar/2_train/2_ce_infer.py
```

***2-Diversity Rewarding:*** After obtaining the classification softmax scores, we calculate the ground truth for entropy. Then, we use an MLP trained with MSE loss to perform regression, enabling the model to learn an understanding of data entropy. Finally, we predict the entropy for each individual piece of the original data.

```bash
# Convert domain softmax to entropy
python ./daar/2_train/3_scores2entropy.py

# Training entropy-probe
python ./daar/2_train/4_mse_train.py

# Infering on raw data
python ./daar/2_train/5_mse_infer.py
```

***3-Data Selection:*** Finally, based on the predicted entropy scores, we select the top 20% of the data to obtain the DaaR training data in `./daar/2_train/mse_res/[models]/daar_data.jsonl`

```bash
# Data selection
python ./daar/2_train/6_select_data.py
```

## Implementation of Baselines

### 1. Random Selection

Random sampling refers to randomly selecting 20% from different domains to simulate an average distribution. The execution code is as:

```bash
python ./baselines/rand.py --input_file_path ./data/raw/40k_data.jsonl --input_tsne_path ./data/ebd/qw25/tsne_data.npz --output_path ./data/res/qw25/baselines/rand
```

### 2. Instruction Length & Output Length

Following previous research (mentioned in DEITA), the length of the instruction and output in the data can reliably represent the complexity and quality of the data. Therefore, we use the following code to select the longest 20%:

```bash
python ./baselines/instruction_len.py
```

### 3. Alpagasus

We provide our [Alpagasus](https://github.com/Lichang-Chen/AlpaGasus) scoring prompt and files as follows:

```bash
python ./baselines/alpagasus.py
```

### 4. SuperFiltering

We strictly follow the [SuperFiltering](https://github.com/tianyi-lab/Superfiltering) method for scoring and select the top 20% with the highest scores for data selection. 

### 5. Instag

We follow the [Instag](https://github.com/OFA-Sys/InsTag) concept for scoring. In the original method, ChatGPT was used as a tagger to label the data, which was then trained in its own *InsTagger*. However, we did not use ***InsTagger***; instead, we used GPT-3.5-Turbo as the tagger to obtain the best possible labels. The relevant reference codes for ***Instag-C*** and ***Instag-D*** are as follows:

```bash
# Tag Step
python ./baselines/instag_tagger.py

# Select Step
python ./baselines/instag_select.py
```

### 6. Deita

We strictly follow the [Deita](https://github.com/hkust-nlp/deita) method for scoring and select the top 20% with the highest scores for ***Deita-C***, ***Deita-D*** and ***Deita-Q***. 

## Train

Training can be directly performed using the cli `swift`. Detailed parameter settings can be referred to in the [instruction of ms-swift](https://swift2x-en.readthedocs.io/en/latest/). Below is a sample configuration, the training parameters in the paper are **consistent with this example**:

```bash
bash train/example.sh
```



## Evaluation

Following the [instruction of OpenCompass](https://opencompass.readthedocs.io/en/latest/get\_started/quick\_start.html), we need to enter the `./eval` directory and then **clone the OpenCompass repository** and download its evaluation files:

```bash
cd eval
git clone https://github.com/open-compass/opencompass.git
cd opencompass

# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

Next, you can evaluate specific models and their corresponding checkpoints (please fill in as needed) on seven evaluation tasks: `NQ`, `TriviaQA`, `HellaSwag`, `GSM8K`, `MATH`, `MBPP`, and `HumanEval`.

```bash
python run.py \
    --datasets nq_gen triviaqa_gen hellaswag_ppl gsm8k_gen math_gen mbpp_gen humaneval_gen \
    --hf-type base \
    --hf-path ../../models/Qwen2.5-7B \
    --peft-path ../../ckpt/example \
    --batch-size 16 \
    --hf-num-gpus 1 \
    --debug
```

For this paper, considering the large number of experiments and time constraints, we uniformly tailored the evaluation datasets while maintaining consistency across all evaluations. The number of samples in each dataset is shown in the table below. Specifically, for each original evaluation dataset, we used `df.sample()` with `seed=42` for tailoring.


| Number of Samples | NQ           | Triviaqa  | Hellaswag   | GSM8K  | MATH  | MBPP  | HumanEval |
| :----:            | :----:       | :----:    | :----:      | :----: | :----:| :----:| :----:    |
| Original          | 3,610        | 8,837     | 10,042      | 1,319  | 5,000 | 974   | 164       |
| Utilized          | 3,610        | 5,000     | 10,042      | 500    | 1,000 | 500   | 164       |

## References
If you find our work useful for your research or development, please kindly cite the following [paper](https://www.arxiv.org/abs/2502.04380).
```
@article{ling2025diversity,
  title={Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data},
  author={Ling, Zhenqing and Chen, Daoyuan and Yao, Liuyi and Li, Yaliang and Shen, Ying},
  journal={arXiv preprint arXiv:2502.04380},
  year={2025}
}
```