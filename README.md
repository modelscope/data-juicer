# Official repo for **Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data**

We release **DaaR**,  A data diversity-driven reward method for high-quality data selection across mixed domains to enhance LLM capabilities. See more details in our [paper](https://www.arxiv.org/abs/2502.04380).

> 
>
> **Abstract:** Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains.
In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance.
To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. 
Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data.
Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.


## Codes and Data Recipes

- The original codes are organized and presented in [DaaR](https://github.com/modelscope/data-juicer/tree/DaaR/DaaR).
- We are developing a series of data-juicer operators related to DaaR. Stay tuned.


## Citation

If you find our work useful for your research, please consider citing our paper.

```
@article{ling2025diversity,
  title={Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data},
  author={Ling, Zhenqing and Chen, Daoyuan and Yao, Liuyi and Li, Yaliang and Shen, Ying},
  journal={arXiv preprint arXiv:2502.04380},
  year={2025}
}
```
