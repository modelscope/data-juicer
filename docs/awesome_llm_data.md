# Awesome Data-Model Co-Development of MLLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Welcome to the "Awesome List" for data-model co-development of Multi-Modal Large Language Models (MLLMs), a continually updated resource tailored for the open-source community. This compilation features cutting-edge research, insightful articles focusing on improving MLLMs involving with the data-model co-development of MLLMs, and  tagged based on the proposed **taxonomy** from our data-model co-development [survey](https://arxiv.org/abs/2407.08583), as illustrated below.

![Overview of Our Taxonomy](https://img.alicdn.com/imgextra/i1/O1CN01aN3TVo1mgGZAuSHJ4_!!6000000004983-2-tps-3255-1327.png)
Soon we will provide a dynamic table of contents to help readers more easily navigate through the materials with features such as search, filter, and sort.

Due to the rapid development in the field, this repository and our paper are continuously being updated and synchronized with each other. Please feel free to make pull requests or open issues to [contribute to](#contribution-to-this-survey) this list and add more related resources!



## Candidate Co-Development Tags
These tags correspond to the taxonomy in our paper, and each work may be assigned with more than one tags.
### Data4Model: Scaling
#### For Scaling Up of MLLMs: Larger Datasets
| Section Title | Tag |
|-------|-------|
|Data Acquisition|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|Data Augmentation|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|Data Diversity|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|

#### For Scaling Effectiveness of MLLMs: Better Subsets
| Section Title | Tag |
|-------|-------|
|Data Condensation|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Data Mixture|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|Data Packing|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|Cross-Modal Alignment|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|

### Data4Model: Usability
#### For Instruction Responsiveness of MLLMs
| Section Title | Tag |
|-------|-------|
|Prompt Design|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|ICL Data|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|Human-Behavior Alignment Data|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|

#### For Reasoning Ability of MLLMs
| Section Title | Tag |
|-------|-------|
|Data for Single-Hop Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa)|
|Data for Multi-Hop Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|

#### For Ethics of MLLMs
| Section Title | Tag |
|-------|-------|
|Data Toxicity|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|Data Privacy and Intellectual Property|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Privacy&IP-d3f0aa)|

#### For Evaluation of MLLMs
| Section Title | Tag |
|-------|-------|
|Benchmarks for Multi-Modal Understanding | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|Benchmarks for Multi-Modal Generation: | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|Benchmarks for Multi-Modal Retrieval: | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|
|Benchmarks for Multi-Modal Reasoning: | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|

### Model4Data: Synthesis
| Section Title | Tag |
|-------|-------|
|Model as a Data Creator|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Model as a Data Mapper|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Model as a Data Filter|![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)|
|Model as a Data Evaluator|![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|

### Model4Data: Insights
| Section Title | Tag |
|-------|-------|
|Model as a Data Navigator|![](https://img.shields.io/badge/Model4Data--Insights--Navigator-f2c0c6)|
|Model as a Data Extractor|![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)|
|Model as a Data Analyzer|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|
|Model as a Data Visualizer|![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|


## Paper List
| Title | Tags | Back Reference (In Paper)|
|-------|-------|-------|
|No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|Sec. 1, Sec. 3.1, Sec. 3.1.1, Sec. 3.1.3, Sec. 3.2, Sec. 3.2.4, Sec. 6.2, Sec. 8.2.1, Table 2|
|What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 5.1|
|Med-MMHL: A Multi-Modal Dataset for Detecting Human- and LLM-Generated Misinformation in the Medical Domain|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|Sec. 4.3.1|
|Probing Heterogeneous Pretraining Datasets with Small Curated Datasets|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)||
|ChartLlama: A Multimodal LLM for Chart Understanding and Generation|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|Sec. 5.1, Sec. 6.3, Sec. 6.4|
|VideoChat: Chat-Centric Video Understanding|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 5.1, Sec. 5.2|
|Aligned with LLM: a new multi-modal training paradigm for encoding fMRI activity in visual cortex|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 5.2|
|3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 5.1|
|GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 3.1.1, Sec. 5.2|
|Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1|
|Audio Retrieval with WavText5K and CLAP Training|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|Sec. 3.1.1, Sec. 3.1.3, Sec. 4.4.3|
|The Devil is in the Details: A Deep Dive into the Rabbit Hole of Data Filtering|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 5.3, Sec. 8.3.3|
|Demystifying CLIP Data|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|3.2.2|
|Learning Transferable Visual Models From Natural Language Supervision|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 2.1, Sec. 3.1.1, Sec. 3.2.2|
|DataComp: In search of the next generation of multimodal datasets|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)|Sec. 1, Sec. 3.1.1, Sec. 3.1.3, Sec. 3.2, Sec. 3.2.1, Sec. 3.2.4, Sec. 4.4.1, Sec. 5.3, Sec. 8.1, Sec. 8.3.3, Table 2|
|Beyond neural scaling laws: beating power law scaling via data pruning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1|
|Flamingo: a visual language model for few-shot learning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|Sec. 3.1.3, Sec. 3.2.2|
|Quality not quantity: On the interaction between dataset design and robustness of clip|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|Sec. 3.2.2|
|VBench: Comprehensive Benchmark Suite for Video Generative Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|Sec. 4.4.2|
|EvalCraftr: Benchmarking and Evaluating Large Video Generation Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|Sec. 4.4.2|
|Training Compute-Optimal Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1|
|NExT-GPT: Any-to-Any Multimodal LLM|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 1, Sec. 2.1, Sec. 3.1.1|
|ChartThinker: A Contextual Chain-of-Thought Approach to Optimized Chart Summarization|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|Sec. 3.1.1, Sec. 3.2.4|
|ChartReformer: Natural Language-Driven Chart Image Editing|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|Sec. 3.1.1, Sec. 6.4|
|GroundingGPT: Language Enhanced Multi-modal Grounding Model|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|Sec. 4.1.2|
|Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|Sec. 4.1.1|
|Kosmos-2: Grounding Multimodal Large Language Models to the World|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|Sec. 4.1.1|
|Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters|![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 3.2.1, Sec. 5.1, Sec. 5.3, Sec. 8.3.3|
|Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 8.3.3|
|Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 3.1.1, Sec. 3.1.3, Sec. 4.1.3, Sec. 5.1, Sec. 5.4, Sec. 8.2.3, Sec. 8.3.3, Sec. 8.3.4|
|3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.4.1|
|Structured Packing in LLM Training Improves Long Context Utilization|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|Sec. 3.2.3|
|Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|Sec. 3.2.3|
|MoDE: CLIP Data Experts via Clustering|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|Sec. 3.2.3|
|Efficient Multimodal Learning from Data-centric Perspective|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 1, Sec. 2.1, Sec. 3.2.1|
|Improved Baselines for Data-efficient Perceptual Augmentation of LLMs|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|Sec. 3.1.2|
|MVBench: A Comprehensive Multi-modal Video Understanding Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.4.1|
|SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.4.1|
|Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1|
|Perception Test: A Diagnostic Benchmark for Multimodal Video Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.4.2|
|FunQA: Towards Surprising Video ComprehensionFunQA: Towards Surprising Video Comprehension|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|Sec. 4.2.1, Sec. 4.4.4|
|OneChart: Purify the Chart Structural Extraction via One Auxiliary Token|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 4.4.1, Sec. 5.1, Sec. 6.3|
|ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|Sec. 4.4.4, Sec. 6.3|
|StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa)|Sec. 3.1.1, Sec. 4.2.1, Sec. 6.3|
|MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 3.1.1, Sec. 4.4.1|
|ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|Sec. 3.1.3, Sec. 4.4.4, Sec. 5.1, Sec. 6.3|
|WorldGPT: Empowering LLM as Multimodal World Model|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|Sec. 4.4.2|
|List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|Sec. 3.1.1, Sec. 3.2.2, Sec. 4.1.2|
|TextSquare: Scaling up Text-Centric Visual Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|Sec. 3.1.1, Sec. 5.1, Sec. 5.3, Sec. 5.4, Sec. 8.3.3, Table 2|
|ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1, Sec. 4.4.1|
|How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Insights--Navigator-f2c0c6)|Sec 6.1|
|Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 4.1.1|
|Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|Sec. 3.2.3|
|Fewer Truncations Improve Language Modeling|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|Sec. 3.2.3|
|MedThink: Explaining Medical Visual Question Answering via Multimodal Decision-Making Rationale|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 4.2.2, Sec. 5.2|
|AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 5.2|
|UNIAA: A Unified Multi-modal Image Aesthetic Data AugmentationAssessment Baseline and Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 4.4.1, Sec. 5.1|
|Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 3.1.2, Sec. 5.1|
|Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|Sec. 4.1.1, Sec. 4.3.1, Sec. 5.4|
|TextHawk: Exploring Efficient Fine-Grained Perception of Multimodal Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1|
|The Wolf Within: Covert Injection of Malice into MLLM Societies via an MLLM Operative|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|Sec. 4.3.1|
|BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1, Sec. 5.2|
|MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|Sec. 4.1.3, Sec. 4.4.2, Sec. 5.4, Sec. 8.2.3, Table 2|
|MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)| Sec. 4.3.1,  Sec. 4.4.2|
|Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|Sec. 3.1.3, Sec. 4.1.2, Sec.4.2.2|
|M3DBench: Let’s Instruct Large Models with Multi-modal 3D Prompts|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.4.1|
|MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 5.2|
|mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|Sec. 3.1.2, Sec. 6.3|
|mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|Sec. 6.3|
|mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|Sec 3.1.2|
|mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|Sec. 6.3|
|Open-TransMind: A New Baseline and Benchmark for 1st Foundation Model Challenge of Intelligent Transportation|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|Sec. 4.4.1, Sec. 4.4.3|
|On the Adversarial Robustness of Multi-Modal Foundation Models|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|Sec 4.3.1|
|What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 4.2.1, Sec. 5.1, Sec. 5.3|
|ShareGPT4V: Improving Large Multi-Modal Models with Better Captions|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1|
|PaLM-E: An Embodied Multimodal Language Model|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|Sec. 3.1.3|
|Multimodal Data Curation via Object Detection and Filter Ensembles|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 3.2.4, Sec. 8.3.3|
|Sieve: Multimodal Dataset Pruning Using Image Captioning Models|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 3.2.4, Sec. 8.3.1, Sec. 8.3.3|
|Towards a statistical theory of data selection under weak supervision|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 5.3|
|𝐷2 Pruning: Message Passing for Balancing Diversity & Difficulty in Data Pruning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.3|
|UIClip: A Data-driven Model for Assessing User Interface Design|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 3.1.1|
|CapsFusion: Rethinking Image-Text Data at Scale|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|Sec. 3.1.2|
|Improving CLIP Training with Language Rewrites|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|Sec. 1, Sec. 3.1.2, Sec. 5.2|
|OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|Sec. 4.4.2|
|A Decade's Battle on Dataset Bias: Are We There Yet?|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|Sec. 3.2.2|
|Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|Sec 3.2.4|
|Data Filtering Networks|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 3.2.4, Sec. 8.3.3|
|T-MARS: Improving Visual Representations by Circumventing Text Feature Learning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 3.2.4, Sec. 8.3.3|
|InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1|
|Align and Attend: Multimodal Summarization with Dual Contrastive Losses|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|Table 2|
|MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|Table 2|
|Text-centric Alignment for Multi-Modality Learning|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 3.2.4|
|Noisy Correspondence Learning with Meta Similarity Correction|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|Sec. 3.2.4|
|Grounding-Prompter: Prompting LLM with Multimodal Information for Temporal Sentence Grounding in Long Videos|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|Sec. 4.2.2|
|Language-Image Models with 3D Understanding|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|Sec. 4.2.2|
|Scaling Laws for Generative Mixed-Modal Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 1|
|BLINK: Multimodal Large Language Models Can See but Not Perceive|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.4.1, Table 2|
|Visual Hallucinations of Multi-modal Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|Sec. 4.4.2, Sec. 5.3|
|DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|Sec. 4.2.2|
|EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 3.1.1, Sec. 4.2.2, Sec. 5.1|
|Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|Sec. 3.1.1, Sec. 4.2.2, Table 2|
|Visual Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 3.1.1|
|ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 2.1, Sec. 3.1.1, Sec. 3.2.4, Sec. 4.1, Sec. 4.1.1, Sec. 4.1.3, Sec. 8.3.1, Table 2|
|Time-LLM: Time Series Forecasting by Reprogramming Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|Sec. 4.1.1|
|On the De-duplication of LAION-2B|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec 3.2.1|
|Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|Sec. 3.1.1, Sec. 3.2.2|
|LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|Sec. 4.1.3, Sec. 4.4.1, Table 2|
|LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|Sec. 6.2|
|Data Augmentation for Text-based Person Retrieval Using Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 3.1.2, Sec. 5.2|
|Aligning Actions and Walking to LLM-Generated Textual Descriptions|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 3.1.2, Sec. 5.2|
|GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|Sec. 3.1.2|
|SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|Sec. 3.1.3|
|AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|3.2.4|
|AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 5.1|
|Probing Multimodal LLMs as World Models for Driving|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|Sec. 3.1.1, Sec. 4.4.4|
|Unified Hallucination Detection for Multimodal Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 4.4.2, Sec. 5.2, Sec. 6.2, Table 2|
|Semdedup: Data-efficient learning at web-scale through semantic deduplication|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|Sec. 3.2.1, Sec. 3.2.4, Sec. 8.3.3|
|Automated Multi-level Preference for MLLMs|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 4.1.3|
|Silkie: Preference distillation for large visual language models|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 4.1.3|
|Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 4.1.3, Table 2|
|M3it: A large-scale dataset towards multi-modal multilingual instruction tuning|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Table 2|
|Aligning Large Multimodal Models with Factually Augmented RLHF|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 4.1.3|
|DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|Sec. 4.1.3|
|RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|Sec. 4.1.3|
|MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|Sec. 4.4.2, Sec. 5.4, Sec. 8.3.3, Sec. 8.3.4, Table 2|
|MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|Sec. 4.4.3, Table 2|
|M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|Sec. 4.4.4, Table 2|
|ImgTrojan: Jailbreaking Vision-Language Models with ONE Image|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 4.3.1, Sec. 5.4|
|VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|Sec. 4.3.1|
|Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|Sec. 4.3.1|
|Improving Multimodal Datasets with Image Captioning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|3.2.1, 3.2.4, 8.2.2, 8.3.3|
|Bridging Research and Readers: A Multi-Modal Automated Academic Papers Interpretation System|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|6.3|
|LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition|![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)|6.2|
|PDFChatAnnotator: A Human-LLM Collaborative Multi-Modal Data Annotation Tool for PDF-Format Catalogs|![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|Sec. 5.2, Sec. 6.2|
|CiT: Curation in Training for Effective Vision-Language Data|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|Sec. 2.1, Sec. 8.3.3|
|InstructPix2Pix: Learning to Follow Image Editing Instructions|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|Sec. 5.1|
|Automated Data Visualization from Natural Language via Large Language Models: An Exploratory Study|![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|Sec. 6.4|
|ModelGo: A Practical Tool for Machine Learning License Analysis|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Privacy&IP-d3f0aa)|Sec. 4.3.2, Sec. 8.2.1|
|Scaling Laws of Synthetic Images for Model Training ... for Now|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|Sec 4.1.1|
|Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|Sec. 3.1.3|
|Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|Sec. 4.1.1|
|Segment Anything|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Sec. 1, Sec. 8.3.1|
|AIM: Let Any Multi-modal Large Language Models Embrace Efficient In-Context Learning|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|Sec 4.1.2|
|MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|Sec 4.1.2|
|All in an Aggregated Image for In-Image Learning|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|Sec. 4.1.2|
|Panda-70m: Captioning 70m videos with multiple cross-modality teachers|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Table 2|
|Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved With Text|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Table 2|
|ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|Table 2|

# Contribution to This Survey
Due to the rapid development in the field, this repository and our paper are continuously being updated and synchronized with each other. Please feel free to make pull requests or open issues to contribute to this list and add more related resources!
You can add the titles of relevant papers to the table above, and (optionally) provide suggested tags along with the corresponding sections if possible.

# References
If you find our work useful for your research or development, please kindly cite the following [paper](https://arxiv.org/abs/2407.08583).


```
@article{qin2024synergy,
  title={The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective},
  author={Qin, Zhen and Chen, Daoyuan and Zhang, Wenhao and Liuyi, Yao and Yilun, Huang and Ding, Bolin and Li, Yaliang and Deng, Shuiguang},
  journal={arXiv preprint arXiv:2407.08583},
  year={2024}
}
```
