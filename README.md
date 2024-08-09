# Official repo for **Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models**

We release **Img-Diff**,  A high-quality synthesis dataset focusing on describing object differences for MLLMs. See more details in our [paper](https://arxiv.org/abs/2408.04594) and download the dataset: [huggingface](https://huggingface.co/datasets/datajuicer/Img-Diff). 

> 
>
> **Abstract:** High-performance Multimodal Large Language Models (MLLMs) rely heavily on data quality. This study introduces a novel dataset named Img-Diff, designed to enhance fine-grained image recognition in MLLMs by leveraging insights from contrastive learning and image difference captioning. By analyzing object differences between similar images, we challenge models to identify both matching and distinct components. We utilize the Stable-Diffusion-XL model and advanced image editing techniques to create pairs of similar images that highlight object replacements. Our methodology includes a Difference Area Generator for object differences identifying, followed by a Difference Captions Generator for detailed difference descriptions. The result is a relatively small but high-quality dataset of "object replacement" samples. We use the the proposed dataset to finetune state-of-the-art (SOTA) MLLMs such as MGM-7B, yielding comprehensive improvements of performance scores over SOTA models that trained with larger-scale datasets, in numerous image difference and Visual Question Answering tasks. For instance, our trained models notably surpass the SOTA models GPT-4V and Gemini on the MMVP benchmark. Besides, we investigate alternative methods for generating image difference data through "object removal" and conduct a thorough evaluation to confirm the dataset's diversity, quality, and robustness, presenting several insights on the synthesis of such a contrastive dataset. We release our codes and dataset, to encourage further research and advance the field of multimodal data synthesis and enhancement of MLLMs' fundamental capabilities for image understanding.


![object_replacement_overview](https://img.alicdn.com/imgextra/i1/O1CN01Ut5eAM1TseaW5g17W_!!6000000002438-2-tps-3970-1778.png)

**Picture**: Illustration of the generation process for “object replacement” data within Img-Diff.




## Codes and Data Recipes

Our codes and data recipes are still being organized and restructured. We will release them in a few days!





## Citation

If you find our work useful for your research, please consider citing our paper.

```
@misc{jiao2024imgdiffcontrastivedatasynthesis,
      title={Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models}, 
      author={Qirui Jiao and Daoyuan Chen and Yilun Huang and Yaliang Li and Ying Shen},
      year={2024},
      eprint={2408.04594},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04594}, 
}
```