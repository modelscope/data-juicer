import os
from pathlib import Path

import jsonlines
import pandas as pd
import streamlit as st
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import Analyser, Executor

demo_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(demo_path))

stack_exchange_recipe_desc = '''
# RedPajama & The Pile -- StackExchange (refined by Data-Juicer)

A refined version of StackExchange dataset in RedPajama & The Pile by [Data-Juicer](https://github.com/alibaba/data-juicer). Removing some "bad" samples from the original merged dataset to make it higher-quality.

This dataset is usually used to pretrain a Large Language Model.

**Notice**: Here is a small subset for previewing. The whole dataset is available [here](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-pile-stackexchange-refine-result.jsonl) (About 71GB).

## Dataset Information

- Number of samples: 26,309,203	 (Keep ~57.89% from the original dataset)

## Refining Recipe
'''

data_juicer_recipe_desc = '''
# Refined open source dataset by Data-Juicer

We found that there are still some "bad" samples in existing processed datasets (e.g. RedPajama, The Pile.). So we use our Data-Juicer to refine them and try to feed them to LLMs for better performance.

We use simple 3-σ rule to set the hyperparameters for ops in each recipe.

## Before and after refining for Pretraining Dataset

| subset               |       #samples before       | #samples after | keep ratio |data link                                                                                                                                                                                                                                                                                  | source                  |
|----------------------|:---------------------------:|:--------------:|:----------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Arxiv                |          1,724,497          |   1,655,259    |   95.99%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-arxiv-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-arxiv-refined-by-data-juicer/summary)                                       | Redpajama               |
| Books                |           205,182           |    195,983     |   95.51%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-book-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-book-refined-by-data-juicer/summary)                                         | Redpajama               |
| Wikipedia            |         29,834,171          |   26,990,659   |   90.47%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-wiki-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-wiki-refined-by-data-juicer/summary)                                         | Redpajama               |
| C4                   |         364,868,892         |  344,491,171   |   94.42%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-c4-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-c4-refined-by-data-juicer/summary)                                             | Redpajama               |
| Common Crawl 2019-30 |         81,085,420          |   36,557,283   |   45.08%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2019-30-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2019-30-refined-by-data-juicer/summary) | Redpajama               |
| Common Crawl 2020-05 |         90,850,492          |   42,612,596   |   46.90%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2020-05-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2020-05-refined-by-data-juicer/summary) | Redpajama               |
| Common Crawl 2021-04 |         98,878,523          |   44,724,752   |   45.23%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2021-04-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2021-04-refined-by-data-juicer/summary) | Redpajama               |
| Common Crawl 2022-05 |         94,058,868          |   42,648,496   |   45.34%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2022-05-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2022-05-refined-by-data-juicer/summary) | Redpajama               |
| Common Crawl 2023-06 |         111,402,716         |   50,643,699   |   45.46%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2023-06-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2023-06-refined-by-data-juicer/summary) | Redpajama               |
| Github Code          | 73,208,524 <br>+ 21,387,703 |   49,279,344   |   52.09%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-stack-code-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-stack-code-refined-by-data-juicer/summary)                             | Redpajama<br>The Stack  |
| StackExchange        |         45,447,328          |   26,309,203   |   57.89%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-pile-stackexchange-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-pile-stackexchange-refined-by-data-juicer/summary)             | Redpajama<br>The Pile   |
| EuroParl             |           69,814            |     61,601     |   88.23%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-europarl-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-europarl-refined-by-data-juicer/summary)                                   | The Pile                |
| FreeLaw              |          3,562,015          |   2,942,612    |   82.61%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-freelaw-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-freelaw-refined-by-data-juicer/summary)                                     | The Pile                |
| HackerNews           |           373,027           |    371,331     |   99.55%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-hackernews-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-hackernews-refined-by-data-juicer/summary)                               | The Pile                |
| NIH ExPorter         |           939,661           |    858,492     |   91.36%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-hin-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-nih-refined-by-data-juicer/summary)                                             | The Pile                |
| PhilPapers           |           32,782            |     29,117     |   88.82%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-philpaper-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-philpaper-refined-by-data-juicer/summary)                                 | The Pile                |
| PubMed Abstracts     |         15,518,009          |   15,009,325   |   96.72%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-pubmed-abstract-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-pubmed-abstracts-refined-by-data-juicer/summary)                    | The Pile                |
| PubMed Central       |          3,098,930          |   2,694,860    |   86.96%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-pubmed-central-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-pubmed-central-refined-by-data-juicer/summary)                       | The Pile                |
| USPTO                |          5,883,024          |   4,516,283    |   76.77%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-uspto-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-uspto-refined-by-data-juicer/summary)                                         | The Pile                |


## Before and after refining for Alpaca-CoT Dataset

| subset               |       #samples before       | #samples after | keep ratio |data link                                                                                                                                                                                                                                                                                  | source                  |
|----------------------|:---------------------------:|:--------------:|:----------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Alpaca-Cot EN | 136,219,879               | 72,855,345 |   54.48% | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/SFT/alpaca-cot-en-refine_result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/alpaca-cot-en-refined-by-data-juicer/summary)                           | [39 Subsets of Alpaca-CoT](alpaca_cot/README.md#refined-alpaca-cot-dataset-meta-info)              |
| Alpaca-Cot ZH | 21,197,246               |               9,873,214                |   46.58%   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/SFT/alpaca-cot-zh-refine_result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/alpaca-cot-zh-refined-by-data-juicer/summary)                           | [28 Subsets of Alpaca-CoT](alpaca_cot/README.md#refined-alpaca-cot-dataset-meta-info)              |
'''


@st.cache_data
def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(encoding='utf_8_sig').encode('utf-8')


@st.cache_data
def convert_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf-8')


def process_and_show_res():

    config_file = os.path.join(
        project_path,
        'configs/data_juicer_recipes/redpajama-pile-stackexchange-refine.yaml')
    dataset_path = os.path.join(demo_path, 'data/stack_exchange.jsonl')
    export_path = os.path.join(demo_path,
                               'outputs/processed_stack_exchange.jsonl')
    cfg_cmd = f'--config {config_file} --dataset_path {dataset_path} --export_path {export_path}'
    args_in_cmd = cfg_cmd.split()
    cfg = init_configs(args=args_in_cmd)
    cfg.open_tracer = True
    cfg.np = 1
    logger.info('=========Stage 1: analyze original data=========')
    analyzer = Analyser(cfg)
    analyzed_dataset = analyzer.run()

    logger.info('=========Stage 2: process original data=========')
    executor = Executor(cfg)
    processed_dataset = executor.run()
    st.session_state.analyzed_dataset = analyzed_dataset
    st.session_state.processed_dataset = processed_dataset

    trace_dir = executor.tracer.work_dir
    trace_files = list(Path(trace_dir).glob('*jsonl'))
    st.session_state.trace_files = trace_files


class Visualize:

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Data-Juicer',
            page_icon=':smile',
            layout='wide',
            # initial_sidebar_state="expanded",
        )

        readme_link = 'https://github.com/alibaba/data-juicer'
        st.markdown(
            '<div align = "center"> <font size = "70"> Data-Juicer \
            </font> </div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div align = "center"> A Data-Centric Text Processing System for \
                Large Language Models, \
                see more details in our <a href={readme_link}>page</a></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def show_recipe():

        def show_yaml(config_file):
            with open(config_file, 'r') as f:
                st.code(f.read(), language='yaml', line_numbers=False)

        with st.expander('Data-Juicer-stack-exchange Recipe', expanded=False):
            st.markdown(stack_exchange_recipe_desc)
            config_file = os.path.join(
                project_path,
                'configs/data_juicer_recipes/redpajama-pile-stackexchange-refine.yaml'
            )
            show_yaml(config_file)

    @staticmethod
    def analyze_process():

        start_btn_process = st.button('Start to process data',
                                      use_container_width=True)

        with st.expander('Data Processing Results', expanded=True):

            if start_btn_process:
                with st.spinner('Wait for process...'):
                    process_and_show_res()

            col1, col2 = st.columns(2)
            with col1:
                st.header('Original Data')
                analyzed_dataset = st.session_state.get(
                    'analyzed_dataset', None)
                st.dataframe(analyzed_dataset, use_container_width=True)
                st.download_button('Download Original data as JSONL',
                                   data=convert_jsonl(
                                       pd.DataFrame(analyzed_dataset)),
                                   file_name='original_dataset.jsonl')

            with col2:
                st.header('Processed Data')
                processed_dataset = st.session_state.get(
                    'processed_dataset', None)
                st.dataframe(processed_dataset, use_container_width=True)
                st.download_button('Download Processed data as JSONL',
                                   data=convert_jsonl(
                                       pd.DataFrame(processed_dataset)),
                                   file_name='processed_dataset.jsonl')

            trace_files = st.session_state.get('trace_files', [])

            def display_tracer_result(op_type, prefix, files):
                st.subheader(op_type)
                for file in files:

                    filename = file.stem
                    filepath = str(file)

                    if filename.startswith(prefix):
                        st.markdown(f'- {filename.split(prefix)[1]}')
                        with jsonlines.open(filepath, 'r') as reader:
                            objs = [obj for obj in reader]
                        for obj in objs:
                            # simhash value may exceed the range of
                            # integer type of streamlit
                            if 'simhash_deduplicator' in filename:
                                obj['dup1'].pop('simhash')
                                obj['dup2'].pop('simhash')

                        st.dataframe(objs)

            if len(trace_files) > 0:

                st.header('Tracer Results')
                for op_type, prefix in zip(
                    ['Mapper', 'Filter', 'Deduplicator'],
                    ['mapper-', 'filter-', 'duplicate-']):
                    display_tracer_result(op_type, prefix, trace_files)
        with st.expander('Refined Datasets by Data-Juicer', expanded=False):
            st.markdown(data_juicer_recipe_desc)

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.show_recipe()
        Visualize.analyze_process()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
