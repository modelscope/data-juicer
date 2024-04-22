# Awesome LLM Data [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Welcome to the "Awesome List" for LLM-Data, a continually updated resource tailored for the open-source community. This compilation features cutting-edge research, insightful articles, datasets, tools, and frameworks focusing on the nuances of data preparation, processing, enhancement, evaluation and understanding for Large Language Models (LLMs).

We provide a tag-based categorization to help readers easy diving into the myriad of materials, promoting an intuitive understanding of each entry's key focus areas. Soon we will provide a dynamic table of contents to help readers more easily navigate through the materials with features such as search, filter, and sort.

Explore, contribute, and stay up to date with the evolving landscape of LLM-Data. Please feel free to pull requests or open issues to improve this list and add more related resources!

## Tags for the Materials
Material Type
- preprint_publication     # e.g., arXiv'2305 indicates 2023-05 announced
- conference_or_journal_paper 	  # e.g., ACL'23, NeurIPS'23, ...
- Blog_Post
- Tool_Resource
- Dataset_Release
- Framework_Development
- Competition_Challenge

Dataset and Data Type
- Data_Usage_Pretrain
- Data_Usage_FineTune
- Data_Usage_Evaluation
- Data_Domain_Text
- Data_Domain_Multimodal
- Data_Domain_Vision
- Data_Domain_Audio
- Data_Domain_Video
- Data_Domain_Code
- Data_Domain_Web
- Data_Domain_Prompt

Data Understanding 
- Data_Quality
- Data_Diversity
- Data_Quantity
- Data_Contamination
- Data_Bias
- Data_Toxicity
- Privacy_Risks
- Data_Generalization

Data Management
- Data_Processing_Enhancement
- Data_Processing_Mixture
- Data_Processing_Denoising
- Data_Processing_Deduplication
- Data_Processing_Selection
- Data_Curation_RuleBased
- Data_Curation_ModelBased
- Data_Alignment
- Data_Scaling

## Material List

| Material Full Name                                                                                                      | Tags                                                                                                                                                                                  |
|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RedPajama-v2                                                                                                            | `Blog_Post`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Deduplication`, `Data_Quality`, `Data_Diversity`                                                         |
| The RefinedWeb Dataset for Falcon LLM                                                                                   | `NeurIPS_Dataset_and_Benchmark_Track'2023`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Enhancement`, `Data_Processing_Deduplication`                                        |
| The Pile: An 800GB Dataset of Diverse Text for Language Modeling                                                        | `arXiv'2101`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Quality`, `Data_Diversity`, `Data_Quantity`                                                                         |
| Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ Tasks                                   | `EMNLP'22`, `Data_Usage_Evaluation`, `Data_Domain_Text`, `Data_Alignment`, `Data_Generalization`                                                                                   |
| LAION-5B: An open large-scale dataset for training next generation image-text models                                    | `NeurIPS'22`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Processing_Enhancement`, `Data_Quantity`                                                                            |
| Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text                                            | `NeurIPS'23`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Processing_Mixture`, `Data_Quantity`, `Data_Diversity`                                                              |
| Data Filtering Networks                                                                                                 | `arXiv'2309`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Selection`, `Data_Quality`                                                                               |
| SIEVE: Multimodal Dataset Pruning Using Image Captioning Models                                                         | `arXiv'2310`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Processing_Denoising`, `Data_Quality`                                                                               |
| SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities                         | `arXiv'2305`, `Framework_Development`, `Data_Usage_FineTune`, `Data_Domain_Audio`, `Data_Alignment`, `Data_Generalization`                                                              |
| Listen, Think, and Understand                                                                                           | `arXiv'2305`, `Framework_Development`, `Data_Usage_FineTune`, `Data_Domain_Audio`, `Data_Diversity`, `Data_Generalization`                                                              |
| AudioCaps: Generating Captions for Audios in The Wild                                                                   | `NAACL'19`, `Data_Usage_FineTune`, `Data_Domain_Audio`, `Data_Processing_Enhancement`, `Data_Diversity`                                                                           |
| WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research             | `arXiv'2303`, `Data_Usage_FineTune`, `Data_Domain_Audio`, `Data_Processing_Denoising`, `Data_Generalization`                                                                        |
| Improving Multimodal Datasets with Image Captioning                                                                     | `NeurIPS'23`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Curation_ModelBased`, `Data_Processing_Enhancement`, `Data_Quality`                                                 |
| Demystifying CLIP Data                                                                                                  | `arXiv'2309`, `Framework_Development`, `Data_Usage_Pretrain`, `Data_Domain_Vision`, `Data_Curation_RuleBased`, `Data_Quantity`                                                          |
| The Flan Collection: Designing Data and Methods for Effective Instruction Tuning                                        | `ICML'23`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Generalization`                                                                                   |
| Data-Juicer: A One-Stop Data Processing System for Large Language Models                                                | `SIGMOD'24`, `Tool_Resource`, `Framework_Development` `Data_Usage_Pretrain`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Processing_Enhancement`, `Data_Scaling`, `Data_Quality` |
| From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning               | `NAACL'24`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Quality`, `Data_Generalization`                                                                   |
| InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation                                 | `ICLR'24`, `Data_Usage_Pretrain`, `Data_Domain_Video`, `Data_Domain_Multimodal`, `Data_Quality`, `Data_Curation_ModelBased`                                                      |
| What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning             | `ICLR'24`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Quality`, `Data_Diversity`                                                                        |
| Alpagasus: Training a Better Alpaca Model with Fewer Data                                                               | `ICLR'24`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Quality`                                                                                          |
| WaveCoder: Widespread And Versatile Enhanced Instruction Tuning with Refined Data Generation                            | `arXiv'2312`, `Framework_Development`, `Data_Usage_FineTune`, `Data_Domain_Code`, `Data_Alignment`, `Data_Generalization`                                                               |
| IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models                      | `ICLR'24`, `Tool_Resource`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Generalization`                                                                                   |
| LoBaSS: Gauging Learnability in Supervised Fine-tuning Data                                                             | `arXiv'2310`, `Tool_Resource`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Scaling`, `Data_Quality`                                                                        |
| Dynosaur: A Dynamic Growth Paradigm for Instruction-Tuning Data Curation                                                | `EMNLP'23`, `Framework_Development`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Processing_Deduplication`, `Data_Alignment`                                                     |
| Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning                                               | `arXiv'2402`, `Tool_Resource`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Scaling`, `Data_Generalization`                                                                 |
| Rethinking the Instruction Quality: LIFT is What You Need                                                               | `arXiv'2312`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Quality`                                                                                          |
| Scaling Laws and Interpretability of Learning from Repeated Data                                                        | `arXiv'2205`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Contamination`, `Data_Bias`                                                                                         |
| Scaling Data-Constrained Language Models                                                                                | `NeurIPS'24`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Scaling`, `Data_Quantity`                                                                                           |
| To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis                                                | `NeurIPS'24`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Deduplication`, `Data_Generalization`                                                                    |
| D4: Improving llm pretraining via document de-duplication and diversification                                           | `NeurIPS'24`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Curation_RuleBased`, `Data_Generalization`                                                                          |
| Deduplicating training data makes language models better                                                                | `ACL'22`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Diversity`, `Data_Quality`                                                                                                            |
| SemDeDup: Data-efficient learning at web-scale through semantic deduplication                                           | `arXiv'2303`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Deduplication`, `Data_Quantity`                                                                          |
| How Much Do Language Models Copy From Their Training Data? Evaluating Linguistic Novelty in Text Generation Using RAVEN | `TACL'23`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Curation_ModelBased`, `Data_Generalization`                                                                         |
| Deduplicating Training Data Mitigates Privacy Risks in Language Models                                                  | `ICML'22`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Privacy_Risks`, `Data_Processing_Deduplication`                                                                                           |
| PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts                         | `ACL'22`, `Tool_Resource`, `Data_Usage_FineTune`, `Data_Domain_Prompt`, `Data_Processing_Enhancement`                                                                       |
| A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity          | `arXiv'2305`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Toxicity`, `Data_Bias`, `Data_Diversity`                                                                            |
| When less is more: Investigating Data Pruning for Pretraining LLMs at Scale                                             | `arXiv'2309`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Selection`, `Data_Quality`                                                                               |
| Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases   | `arXiv'2303`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Scaling`, `Data_Alignment`                                                                                          |
| Investigating data contamination in modern benchmarks for large language models                                         | `arXiv'2311`, `Data_Usage_Evaluation`, `Data_Domain_Text`, `Data_Contamination`, `Data_Bias`                                                                                         |
| Textbooks Are All You Need                                                                                              | `arXiv'2306`, `Data_Usage_FineTune`, `Data_Domain_Code`, `Data_Processing_Enhancement`, `Data_Quantity`                                                                            |
| Textbooks are all you need ii: phi-1.5 technical report                                                                 | `arXiv'2309`, `Data_Usage_FineTune`, `Data_Domain_Code`, `Data_Generalization`, `Data_Diversity`                                                                                   |
| Quality at a glance: An audit of web-crawled multilingual datasets                                                      | `TACL'22`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Quality`, `Data_Diversity`                                                                                          |
| DataComp: In search of the next generation of multimodal datasets                                                       | `NeurIPS_Dataset_and_Benchmark_Track'2023`, `Competition_Challenge`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Processing_Enhancement`                                            |
| The MiniPile Challenge for Data-Efficient Language Models                                                               | `arXiv'2304`, `Competition_Challenge`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Efficiency`, `Data_Diversity`                                                                   |
| [Contamination Detector for LLMs Evaluation](https://github.com/liyucheng09/Contamination_Detector)                     | `Tool_Resource`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Contamination`                                                                                                      |
| Do Models Really Learn to Follow Instructions? An Empirical Study of Instruction Tuning                                 | `ACL'23`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Quality`, `Data_Bias`                                                                                               |
| Did You Read the Instructions? Rethinking the Effectiveness of Task Definitions in Instruction Learning                 | `ACL'23`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Quality`, `Data_Generalization`                                                                                     |
| Exploring Format Consistency for Instruction Tuning                                                                     | `arXiv'2307`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Processing_Enhancement`, `Data_Generalization`                                                    |
| Data-centric Artificial Intelligence: A Survey                                                                          | `arXiv'2303`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Scaling`, `Data_Quality`, `Data_Diversity`                                                                          |
| Data Management For Large Language Models: A Survey                                                                     | `arXiv'2312`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Scaling`, `Data_Quality`, `Data_Generalization`                                                                     |
| awesome-instruction-dataset                                                                                             | `Repo`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Alignment`, `Data_Scaling`, `Data_Generalization`                                                                                     |
| Koala: An Index for Quantifying Overlaps with Pre-training Corpora                                                      | `EMNLP'23 demo`, `Tool_Resource`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Contamination`                                                                                  |
| Detectig Pretraining Data from Large Language Models                                                                    | `arXiv'2310`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Contamination`, `Data_Quantity`                                                                                     |
| Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks | `EMNLP'23`, `Data_Usage_Evaluation`, `Data_Domain_Text`, `Data_Contamination`, `Data_Quality`                                                                                      |
| SlimPajama-DC: Understanding Data Combinations for LLM Training                                                         | `arXiv'2309`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Deduplication`, `Data_Diversity`, `Data_Scaling`                                                         |
| CodeGen2: Lessons for Training LLMs on Programming and Natural Languages                                                | `ICLR'23`, `Data_Usage_FineTune`, `Data_Domain_Code`, `Data_Processing_Mixture`, `Data_Generalization`                                                                          |
| DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining                                                   | `NeurIPS'24`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Processing_Enhancement`, `Data_Domain_Multimodal`, `Data_Scaling`                                                   |
| Oasis: Data Curation and Assessment System for Pretraining of Large Language Models                                     | `arXiv'2311`, `Tool_Resource`, `Data_Usage_Pretrain`, `Data_Processing_Deduplication`, `Data_Quality`, `Data_Processing_Selection`                                              |
| Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace                           | `arXiv'2310`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Processing_Enhancement`, `Data_Generalization`, `Data_Diversity`                                                    |
| How abilities in large language models are affected by supervised fine-tuning data composition                          | `arXiv'2310`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Scaling`, `Data_Generalization`, `Data_Processing_Selection`                                                        |
| Scaling Relationship on Learning Mathematical Reasoning with Large Language Models                                      | `arXiv'2308`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Processing_Deduplication`, `Data_Quantity`, `Data_Generalization`                                                   |
| Data-Centric Financial Large Language Models                                                                            | `arXiv'2310`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Quantity`, `Data_Processing_Enhancement`                                                                            |
| Ziya2: Data-centric Learning is All LLMs Need                                                                           | `arXiv'2311`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Domain_Code`, `Data_Quality`                                                                                        |
| Scaling Laws for Neural Language Models                                                                                 | `arXiv'2001`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Quantity`, `Data_Scaling`                                                                                           |
| Scaling Laws for Autoregressive Generative Modeling                                                                     | `arXiv'2010`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Quantity`, `Data_Scaling`                                                                                           |
| Beyond neural scaling laws: beating power law scaling via data pruning                                                  | `NeurIPS'22`, `Data_Usage_Pretrain`, `Data_Domain_Vision`, `Data_Quantity`, `Data_Quality`, `Data_Scaling`, `Data_Processing_Selection`                                              |
| Reproducible scaling laws for contrastive language-image learning                                                       | `CVPR'23`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Quantity`, `Data_Scaling`                                                                                           |
| An Inverse Scaling Law for CLIP Training                                                                                | `NeurIPS'23`, `Data_Usage_Pretrain`, `Data_Usage_FineTune`, `Data_Domain_Multimodal`, `Data_Quantity`, `Data_Scaling`                                                                 |
| Scale Efficiently: Insights From Pre-Training And Fine-Tuning Transformers                                              | `ICLR'22`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Quantity`, `Data_Scaling`                                                                                                            |
| LIMA: Less Is More for Alignment                                                                                        | `NeurIPS'23`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Quality`, `Data_Quantity`, `Data_Processing_Selection`                                                              |
| LESS: Selecting Influential Data for Targeted Instruction Tuning                                                        | `arXiv'2402`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Quality`, `Data_Processing_Selection`                                                                               |
| Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP                                  | `NeurIPS'22`, `Data_Usage_Pretrain`, `Data_Domain_Multimodal`, `Data_Quality`, `Data_Processing_Mixture`                                                                                 |
| Data Similarity is Not Enough to Explain Language Model Performance                                                     | `EMNLP'23`, `Data_Usage_Pretrain`, `Data_Usage_FineTune`, `Data_Domain_Text`, `Data_Diversity`                                                                                      |
| On the Connection between Pre-training Data Diversity and Fine-tuning Robustness                                        | `NeurIPS'23`, `Data_Usage_Pretrain`, `Data_Usage_FineTune`, `Data_Domain_Vision`, `Data_Domain_Multimodal`, `Data_Diversity`, `Data_Quantity`                                         |
| Data Selection for Language Models via Importance Resampling                                                            | `NeurIPS'23`, `Data_Usage_Pretrain`, `Data_Domain_Text`, `Data_Quality`, `Data_Processing_Selection`                                                                               |
| A Survey on Data Selection for Language Models                                                                          | `arXiv'2403`, `Data_Usage_Pretrain`, `Data_Usage_FineTune`, `Data_Domain_Multimodal`, `Data_Domain_Text`, `Data_Quality`, `Data_Processing_Selection`                                 |
