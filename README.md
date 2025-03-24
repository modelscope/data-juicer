# MindGYM: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions

> Large vision-language models (VLMs) face challenges in achieving robust, transferable reasoning abilities due to reliance on labor-intensive manual instruction datasets or computationally expensive self-supervised methods. To address these issues, we introduce MindGYM, a framework that enhances VLMs through synthetic self-challenging questions, consisting of three stages: (1) Seed Single-Hop Question Synthesis, generating cognitive questions across textual (e.g., logical deduction) and multimodal contexts (e.g., diagram-based queries) spanning eight semantic areas like ethical analysis; (2) Challenging Multi-Hop Question Synthesis, combining seed questions via diverse principles like bridging, visual-textual alignment, to create multi-step problems demanding deeper reasoning; and (3) Thinking-Induced Curriculum Fine-Tuning, a structured pipeline that progressively trains the model from scaffolded reasoning to standalone inference. By leveraging the model's self-synthesis capability, MindGYM achieves high data efficiency (e.g., +16% gains on MathVision-Mini with only 400 samples), computational efficiency (reducing both training and inference costs), and robust generalization across tasks. Extensive evaluations on seven benchmarks demonstrate superior performance over strong baselines, with notable improvements (+15.77% win rates) in reasoning depth and breadth validated via GPT-based scoring. MindGYM underscores the viability of self-challenging for refining VLM capabilities while minimizing human intervention and resource demands. Code and data are released to advance multimodal reasoning research.

![overview](https://github.com/user-attachments/assets/2bd539d8-5afe-4199-b26a-ae376f48d0b4)

## Codes and Data Recipes

- The original codes are organized and presented in [MindGYM](https://github.com/modelscope/data-juicer/tree/MindGYM/MindGYM).
- We are developing a series of data-juicer operators related to MindGYM. Stay tuned.

## Citation

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
