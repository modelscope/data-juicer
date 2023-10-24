import inspect
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from data_juicer.config import init_configs
from data_juicer.core import Analyser
from data_juicer.format.formatter import FORMATTERS
from data_juicer.ops.base_op import OPERATORS

demo_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(demo_path))

project_desc = '''
**Data-Juicer** is a one-stop data processing system to make data higher-quality, juicier, and more digestible for LLMs.
This project is being actively updated and maintained, and we will periodically enhance and add more features and data recipes. We welcome you to join us in promoting LLM data development and research!
'''

features_desc = '''
- **Broad Range of Operators**: Equipped with 50+ core :blue[operators (OPs)], including `Formatters`, `Mappers`, `Filters`, `Deduplicators`, and beyond.

- **Specialized Toolkits**: Feature-rich specialized toolkits such as `Text Quality Classifier`, `Dataset Splitter`, `Analysers`, `Evaluators`, and more that elevate your dataset handling capabilities.

- **Systematic & Reusable**: Empowering users with a systematic library of reusable `config recipes` and `OPs`, designed to function independently of specific datasets, models, or tasks.

- **Data-in-the-loop**: Allowing detailed data analyses with an automated report generation feature for a deeper understanding of your dataset. Coupled with timely multi-dimension automatic evaluation capabilities, it supports a feedback loop at multiple stages in the LLM development process.

- **Comprehensive Processing Recipes**: Offering tens of `pre-built data processing recipes` for pre-training, CFT, en, zh, and more scenarios.

- **User-Friendly Experience**: Designed for simplicity, with `comprehensive documentation`, `easy start guides` and `demo configs`, and intuitive configuration with simple adding/removing OPs from existing configs.

- **Flexible & Extensible**: Accommodating most types of data formats (e.g., jsonl, parquet, csv, ...) and allowing flexible combinations of OPs. Feel free to `implement your own OPs` for customizable data processing.

- **Enhanced Efficiency**: Providing a speedy data processing pipeline requiring less memory, optimized for maximum productivity.
'''

quick_start_desc = '''
### Data Processing

- Run `process_data.py` tool with your config as the argument to process
  your dataset.

```shell
python tools/process_data.py --config configs/demo/process.yaml
```

### Data Analysis
- Run `analyze_data.py` tool with your config as the argument to analyse your dataset.

```shell
python tools/analyze_data.py --config configs/demo/analyser.yaml
```

- **Note:** Analyser only compute stats of Filter ops. So extra Mapper or Deduplicator ops will be ignored in the analysis process.

### Data Visualization

- Run `app.py` tool to visualize your dataset in your browser.

```shell
streamlit run app.py
```
'''

config_desc = '''

Data-Juicer provides some configuration files to allow users to easily understand the configuration methods of various functions and quickly reproduce the processing flow of different datasets.

### Usage

```shell
# To process your dataset.
python tools/process_data.py --config xxx.yaml
# To analyse your dataset.
python tools/analyze_data.py --config xxx.yaml
```
'''

config_all_desc = '''
`config_all.yaml` which includes **all** ops and default arguments. You just need to **remove** ops that you won't use and refine some arguments of ops.
'''

op_desc = '''
The operators in Data-Juicer are categorized into 5 types.

| Type                              | Number | Description                                     |
|-----------------------------------|:------:|-------------------------------------------------|
| Formatter         |   7    | Discovers, loads, and canonicalizes source data |
| Mapper            |   19   | Edits and transforms samples                    |
| Filter            |   16   | Filters out low-quality samples                 |
| Deduplicator      |   3    | Detects and removes duplicate samples           |
| Selector          |   2    | Selects top samples based on ranking            |
'''

op_list_desc = {
    'formatter':
    '''
| Operator          | Domain  |  Lang  | Description                                                        |
|-------------------|---------|--------|--------------------------------------------------------------------|
| remote_formatter  | General | en, zh | Prepares datasets from remote (e.g., HuggingFace)                  |
| csv_formatter     | General | en, zh | Prepares local `.csv` files                                        |
| tsv_formatter     | General | en, zh | Prepares local `.tsv` files                                        |
| json_formatter    | General | en, zh | Prepares local `.json`, `.jsonl`, `.jsonl.zst` files               |
| parquet_formatter | General | en, zh | Prepares local `.parquet` files                                    |
| text_formatter    | General | en, zh | Prepares other local text files ([complete list](../data_juicer/format/text_formatter.py#L63,73)) |
| mixture_formatter | General | en, zh | Handles a mixture of all the supported local file types
''',
    'mapper':
    '''
| Operator                                      | Domain             | Lang   | Description                                                                                                    |
|-----------------------------------------------|--------------------|--------|----------------------------------------------------------------------------------------------------------------|
| clean_copyright_mapper                              | Code               | en, zh | Removes copyright notice at the beginning of code files (:warning: must contain the word *copyright*)          |
| clean_email_mapper                                  | General            | en, zh | Removes email information                                                                                      |
| clean_html_mapper                                   | General            | en, zh | Removes HTML tags and returns plain text of all the nodes                                                      |
| clean_ip_mapper                                     | General            | en, zh | Removes IP addresses                                                                                           |
| clean_links_mapper                                  | General, Code      | en, zh | Removes links, such as those starting with http or ftp                                                         |
| expand_macro_mapper                                 | LaTeX              | en, zh | Expands macros usually defined at the top of TeX documents                                                     |
| fix_unicode_mapper                                  | General            | en, zh | Fixes broken Unicodes (by [ftfy](https://ftfy.readthedocs.io/))                                                |
| nlpaug_en_mapper                                    | General            | en     | Simply augment texts in English based on the `nlpaug` library                                                  | 
| nlpcda_zh_mapper                                    | General            | zh     | Simply augment texts in Chinese based on the `nlpcda` library                                                  | 
| punctuation_normalization_mapper                    | General            | en, zh | Normalizes various Unicode punctuations to their ASCII equivalents                                             |
| remove_bibliography_mapper                          | LaTeX              | en, zh | Removes the bibliography of TeX documents                                                                      |
| remove_comments_mapper                              | LaTeX              | en, zh | Removes the comments of TeX documents                                                                          |
| remove_header_mapper                                | LaTeX              | en, zh | Removes the running headers of TeX documents, e.g., titles, chapter or section numbers/names                   |
| remove_long_words_mapper                            | General            | en, zh | Removes words with length outside the specified range                                                          |
| remove_specific_chars_mapper                        | General            | en, zh | Removes any user-specified characters or substrings                                                            |
| remove_table_text_mapper                            | General, Financial | en     | Detects and removes possible table contents (:warning: relies on regular expression matching and thus fragile) |
| remove_words_with_incorrect_<br />substrings_mapper | General            | en, zh | Removes words containing specified substrings                                                                  |
| sentence_split_mapper                               | General            | en     | Splits and reorganizes sentences according to semantics                                                        |
| whitespace_normalization_mapper                     | General            | en, zh | Normalizes various Unicode whitespaces to the normal ASCII space (U+0020)                                      |
''',
    'filter':
    '''
| Operator                       | Domain  | Lang   | Description                                                                                |
|--------------------------------|---------|--------|--------------------------------------------------------------------------------------------|
| alphanumeric_filter            | General | en, zh | Keeps samples with alphanumeric ratio within the specified range                           |
| average_line_length_filter     | Code    | en, zh | Keeps samples with average line length within the specified range                          |
| character_repetition_filter    | General | en, zh | Keeps samples with char-level n-gram repetition ratio within the specified range           |
| flagged_words_filter           | General | en, zh | Keeps samples with flagged-word ratio below the specified threshold                        |
| language_id_score_filter       | General | en, zh | Keeps samples of the specified language, judged by a predicted confidence score            |
| maximum_line_length_filter     | Code    | en, zh | Keeps samples with maximum line length within the specified range                          |
| perplexity_filter              | General | en, zh | Keeps samples with perplexity score below the specified threshold                          |
| special_characters_filter      | General | en, zh | Keeps samples with special-char ratio within the specified range                           |
| specified_field_filter         | General | en, zh | Filters samples based on field, with value lies in the specified targets                   |
| specified_numeric_field_filter | General | en, zh | Filters samples based on field, with value lies in the specified range (for numeric types) |
| stopwords_filter               | General | en, zh | Keeps samples with stopword ratio above the specified threshold                            |
| suffix_filter                  | General | en, zh | Keeps samples with specified suffixes                                                      |
| text_length_filter             | General | en, zh | Keeps samples with total text length within the specified range                            |
| token_num_filter               | General | en, zh | Keeps samples with token count within the specified range                                  |
| word_num_filter                | General | en, zh | Keeps samples with word count within the specified range                                   |
| word_repetition_filter         | General | en, zh | Keeps samples with word-level n-gram repetition ratio within the specified range           |
''',
    'deduplicator':
    '''
| Operator                      | Domain  | Lang   | Description                                                 |
|-------------------------------|---------|--------|-------------------------------------------------------------|
| document_deduplicator         | General | en, zh | Deduplicate samples at document-level by comparing MD5 hash |
| document_minhash_deduplicator | General | en, zh | Deduplicate samples at document-level using MinHashLSH      |
| document_simhash_deduplicator | General | en, zh | Deduplicate samples at document-level using SimHash         |
''',
    'selector':
    '''
| Operator                           | Domain  | Lang   | Description                                                           |
|------------------------------------|---------|--------|-----------------------------------------------------------------------|
| topk_specified_field_selector      | General | en, zh | Selects top samples by comparing the values of the specified field    |
| frequency_specified_field_selector | General | en, zh | Selects top samples by comparing the frequency of the specified field |
'''
}

demo_desc = '''
- Introduction to Data-Juicer [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
- Data Visualization:
  - Basic Statistics [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  - Lexical Diversity [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  - Operator Effect [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
- Data Processing:
  - Scientific Literature (e.g. [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  - Programming Code (e.g. [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  - Chinese Instruction Data (e.g. [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
- Tool Pool:
  - Dataset Splitting by Language [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  - Quality Classifier for CommonCrawl [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  - Auto Evaluation on [HELM](https://github.com/stanford-crfm/helm) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  - Data Sampling and Mixture [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
- Data Processing Loop [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]
- Data Processing HPO [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_hpo/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_hpo)]
'''


def run_demo():

    config_file = os.path.join(project_path, 'configs/demo/analyser.yaml')
    data_path = os.path.join(demo_path, 'data/demo-dataset.jsonl')
    st.markdown(f'dataset: `{data_path}`')
    start_btn = st.button(' Start to analyze', use_container_width=True)

    cfg_cmd = f'--config {config_file} --dataset_path {data_path}'
    args_in_cmd = cfg_cmd.split()
    cfg = init_configs(args=args_in_cmd)

    images_ori = []
    cfg['save_stats_in_one_file'] = True

    if start_btn:
        analyzer = Analyser(cfg)

        with st.spinner('Wait for analyze...'):
            analyzer.run()

        overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
        analysis_res_ori = pd.DataFrame()
        if os.path.exists(overall_file):
            analysis_res_ori = pd.read_csv(overall_file)
            
        if os.path.exists(analyzer.analysis_path):
            for f_path in os.listdir(analyzer.analysis_path):
                if '.png' in f_path and 'all-stats' in f_path:
                    images_ori.append(os.path.join(analyzer.analysis_path, f_path))

        st.subheader('Statistics')
        st.dataframe(analysis_res_ori, use_container_width=True)
        if len(images_ori) > 0:
            st.subheader('Histograms')
            for img in images_ori:
                st.image(img, output_format='png', use_column_width=True)


class Visualize:

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Data-Juicer',
            page_icon=':smile',
            # layout='wide',
            # initial_sidebar_state="expanded",
        )

        readme_link = 'https://github.com/alibaba/data-juicer'

        st.markdown(
            '<div align = "center"> <font size = "70"> Data-Juicer  \
            </font> </div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div align = "center"> A One-Stop Data Processing System for \
                Large Language Models, \
                see more details in <a href={readme_link}>GitHub</a></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def overrall():

        st.image(os.path.join(project_path, 'docs/imgs/data-juicer.jpg'),
                 output_format='jpg',
                 use_column_width=True)
        st.markdown(project_desc)
        with st.expander('Features', expanded=False):
            st.markdown(features_desc)
        with st.expander('Quick Start', expanded=False):
            st.markdown(quick_start_desc)
        with st.expander('Configuration', expanded=False):
            st.markdown(config_desc)
            st.markdown('### Recipes')
            tab1, tab2, tab3, tab4 = st.tabs([
                ':blue[ Reproduced Redpajama]', ':blue[ Reproduced BLOOM]',
                ':blue[ Data-Juicer Recipes]', ':blue[ Config-all]'
            ])

            def show_yaml(config_file):
                with open(config_file, 'r') as f:
                    st.code(f.read(), language='yaml', line_numbers=False)

            with tab1:
                label = 'Data-Juicer have reproduced the processing \
                         flow of some RedPajama datasets.'

                config_files = Path(
                    os.path.join(
                        project_path,
                        'configs/reproduced_redpajama')).glob('*.yaml')
                config_dict = {
                    config.stem: str(config)
                    for config in config_files
                }

                selected = st.selectbox(
                    label=label,
                    options=sorted(list(config_dict.keys())),
                )
                show_yaml(config_dict[selected])

            with tab2:
                label = 'Data-Juicer have reproduced the processing flow \
                    of some BLOOM datasets.'

                config_files = Path(
                    os.path.join(project_path,
                                 'configs/reproduced_bloom')).glob('*.yaml')
                config_dict = {
                    config.stem: str(config)
                    for config in config_files
                }

                selected = st.selectbox(
                    label=label,
                    options=sorted(list(config_dict.keys())),
                )
                show_yaml(config_dict[selected])
            with tab3:
                label = 'Data-Juicer have refined some open source datasets \
                    (including CFT datasets) by using Data-Juicer and have \
                        provided configuration files for the refine flow.'

                config_files = Path(
                    os.path.join(
                        project_path,
                        'configs/data_juicer_recipes')).rglob('*.yaml')
                config_dict = {
                    config.stem: str(config)
                    for config in config_files
                }

                selected = st.selectbox(
                    label=label,
                    options=sorted(list(config_dict.keys())),
                )
                show_yaml(config_dict[selected])
            with tab4:
                st.markdown(config_all_desc)
                show_yaml(os.path.join(project_path,
                                       'configs/config_all.yaml'))

        with st.expander('Operators', expanded=False):

            st.markdown(op_desc)
            st.markdown('### Operator List')
            tabs = st.tabs([
                ':blue[ Formatters]',
                ':blue[ Mappers]',
                ':blue[ Filters]',
                ':blue[ Deduplicators]',
                ':blue[ Selecttors]',
            ])

            for op_type, tab in zip(
                ['formatter', 'mapper', 'filter', 'deduplicator', 'selector'],
                    tabs):
                with tab:
                    show_list = st.checkbox(f'Show {op_type} list')
                    if show_list:
                        st.markdown(op_list_desc[op_type])
                    if op_type == 'formatter':
                        repo = FORMATTERS
                        options = list(repo.modules.keys())

                    else:
                        repo = OPERATORS
                        all_ops = list(repo.modules.keys())
                        options = [
                            name for name in all_ops if name.endswith(op_type)
                        ]

                    label = f'Select a {op_type} to show details'
                    op_name = st.selectbox(label=label, options=options)

                    op_class = repo.modules[op_name]

                    st.markdown('#### Source Code')
                    text = inspect.getsourcelines(op_class)
                    st.code(''.join(text[0]),
                            language='python',
                            line_numbers=False)
        with st.expander('Demos', expanded=False):
            st.markdown('### Total Demos')
            st.markdown(demo_desc)
            st.markdown('### First Demo: Basic Statistics')
            run_demo()

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.overrall()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
