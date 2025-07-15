import os
import json
import pandas as pd
import streamlit as st
import yaml
from loguru import logger
import random
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datasets import Dataset


from data_juicer.config import init_configs
from data_juicer.core import Analyzer
from data_juicer.ops.base_op import OPERATORS

from prompts import multi_op_prompt
from assistant import consult
from st_operator_pool import StOperatorPool
from attributor import TextEmbdSimilarityAttributor
from copilot_client import call_copilot_service

import time
import threading
import queue


def call_service_in_thread(chat_history, q):
    try:
        response_generator = call_copilot_service(chat_history)
        for chunk in response_generator:
            q.put(chunk)
    finally:
        q.put(None)

def construct_op_dict(json_path="./configs/op_dict.json"):
    import json
    op_dict = {}
    with open(json_path, 'r') as json_file:
        ops = json.load(json_file)
    for op in ops:
        op_dict[op['class_name']] = op
    return op_dict


with open("./configs/default_ops.yaml", "r") as f:
    DEFAULT_OPS = yaml.safe_load(f)

OP_DICT = construct_op_dict()


def downsample(data_path, n=100, seed=0):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    random.seed(seed)
    random.shuffle(data)
    dataset_filename = data_path.split("/")[-1]
    tgt_filename = dataset_filename.split(".")[0] + "_downsampled." + dataset_filename.split(".")[1]
    tgt_dir = "./outputs/tmp"
    os.makedirs(tgt_dir, exist_ok=True)
    tgt_path = os.path.join(tgt_dir, tgt_filename)
    with open(tgt_path, "w") as f:
        for i in range(n):
            f.write(json.dumps(data[i]) + '\n')

    return tgt_path


def word_cloud(data_path):
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line)["text"])

    text = ' '.join(dataset)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()

    stopwords = set(STOPWORDS)
    text = ' '.join(word for word in text.split() if word not in stopwords)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    img_path = "./outputs/tmp/word_cloud.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)

    return img_path


def load_dataset_as_df(path, n=None):
    d = None
    if path.endswith(".jsonl"):
        d = []
        with open(path, "r") as f:
            for line in f:
                d.append(json.loads(line))
        if n:
            d = d[:n]
        return pd.DataFrame(d)
    else:
        raise NotImplementedError("only jsonl files are supported")


def load_dataset(path, n=None):
    d = None
    if path.endswith(".jsonl"):
        d = []
        with open(path, "r") as f:
            for line in f:
                d.append(json.loads(line))
        if n:
            d = d[:n]
        return Dataset.from_list(d)
    else:
        raise NotImplementedError("only jsonl files are supported")


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


def pretty_out(d):
    res = ''
    process = ''
    op_names = set(OPERATORS.modules.keys())
    for key, value in d.items():
        if key == 'process':
            process = yaml.dump(value,
                                allow_unicode=True,
                                default_flow_style=False)
        elif key == 'config' or key.split('.')[0] in op_names:
            continue
        else:
            res += f'{key}:\n \t {value}\n'
    res += 'process:\n' + \
           '\n'.join(['\t' + line for line in process.splitlines()])

    return res


def parse_cfg():
    cfg_file = st.session_state.input_cfg_file
    cfg_cmd = st.session_state.input_cfg_cmd

    cfg_f_name = 'null'
    del_cfg_file = False
    if cfg_file is not None:
        cfg_f_name = cfg_file.name
        file_contents = cfg_file.getvalue()
        with open(cfg_f_name, 'wb') as f:
            f.write(file_contents)
        cfg_cmd = f'--config {cfg_f_name}'
        del_cfg_file = True

    args_in_cmd = cfg_cmd.split()

    if len(args_in_cmd) >= 2 and args_in_cmd[0] == '--config':
        cfg_f_name = args_in_cmd[1]
    else:
        st.warning('Please specify a config command or upload a config file.')
        st.stop()

    if not os.path.exists(cfg_f_name):
        st.warning('do not parse'
                   f'config file does not exist with cfg_f_name={cfg_f_name}')
        st.stop()

    with open(cfg_f_name, 'r') as cfg_f:
        specified_cfg = yaml.safe_load(cfg_f)

    try:
        parsed_cfg = init_configs(args=args_in_cmd)
        st.session_state.cfg = parsed_cfg
        if del_cfg_file:
            os.remove(cfg_f_name)
        return pretty_out(parsed_cfg), pretty_out(specified_cfg), parsed_cfg
    except Exception as e:
        return str(e), pretty_out(specified_cfg), None


def get_dataset_snapshots(n=5, key='text'):
    dataset_path = st.session_state.dataset_path
    dataset_snapshots = []
    if dataset_path.endswith('.jsonl'):
        import json
        with open(dataset_path, 'r') as f:
            for line in f:
                dataset_snapshots.append(json.loads(line)[key])
                if len(dataset_snapshots) >= n:
                    break
        return dataset_snapshots
    else:
        raise ValueError('dataset path must end in .jsonl')


def get_enabled_op_info():
    op_info = {}
    global DEFAULT_OPS
    global OP_DICT
    for op_name, _ in DEFAULT_OPS.items():
        if st.session_state.get(f"{op_name}_enabled", False):
            op_info[op_name] = {"op_desc": OP_DICT[op_name]['class_desc']}
    return op_info

class Visualize:
    op_pool = None

    def __init__(self):
        first_time = st.session_state.get('first_time', True)
        save_path = "./save/op_pool_state.yaml"
        if first_time:
            self.op_pool = StOperatorPool(config_path="./configs/default_ops.yaml")
            # self.op_pool = StOperatorPool(default_ops=default_ops)
            self.op_pool.st_sync()
            st.session_state.op_pool = self.op_pool
            st.session_state.first_time = False
            self.op_pool.save(save_path)
        else:
            self.op_pool = st.session_state.op_pool
            self.op_pool.st_sync()

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Data-Juicer Agent',
            page_icon=':smile',
            layout='wide',
            # initial_sidebar_state="expanded",
        )

        readme_link = 'https://github.com/alibaba/data-juicer'
        st.markdown(
            '<div align = "center"> <font size = "70"> Data-Juicer Agent\
            </font> </div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div align = "center"> A Demonstration for Interactive Recipe Generation Workflow</div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def register():
        with st.expander('Register', expanded=False):
            st.text_area(label='(i) Project Name',
                         key='project_name',
                         value=f'Demo')
            example_data_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             './data/demo-dataset.jsonl'))
            st.text_area(label='(ii) Data Path',
                         key='dataset_path',
                         value=f'{example_data_path}')
            example_task_description = "Train a general purposed language model."
            st.text_area(label='(iii) Task Description',
                         key='task_description',
                         value=f'{example_task_description}')

    def llm_assistant(self):
        with st.expander('LLM Assistant', expanded=False):
            # default_prompt = get_default_prompt()
            # print(self.op_pool[0].state)
            op_states = [op_state for _, op_state in self.op_pool.state.items()]
            default_prompt = multi_op_prompt(
                op_states=op_states,
                task_description=st.session_state.task_description,
                user_prompt=""
            )
            st.text_area(
                label='Prompt',
                key='prompt',
                height=310,
                value=default_prompt
            )
            consult_btn = st.button("Consult", use_container_width=True)
            if consult_btn:
                with st.spinner('Wait for LLM assistant to generate suggestions...'):
                    prompt = st.session_state.prompt
                    suggestions, suggestions_text = consult(prompt)
                    st.session_state.suggestions = suggestions
                    st.write(suggestions_text)
            if st.session_state.get('suggestions', None) is not None:
                col1, col2 = st.columns(2)
                with col1:
                    apply_btn = st.button("Apply", use_container_width=True)
                    if apply_btn:
                        with st.spinner('Applying suggestions...'):
                            for _, s in st.session_state.suggestions.items():
                                op_name, action, arg_name, value = \
                                    s["op_name"], s["action"], s.get("arg_name"), s.get("value")
                                if action == "enable":
                                    self.op_pool[op_name].enable()
                                elif action == "disable":
                                    self.op_pool[op_name].disable()
                                elif action == "modify":
                                    self.op_pool[op_name].args[arg_name].set_v(value)
                                else:
                                    raise ValueError(f"Invalid action {action}")
                            st.session_state.suggestions = None
                with col2:
                    ignore_btn = st.button("Ignore", use_container_width=True)
                    if ignore_btn:
                        st.session_state.suggestions = None


    def analyze_process(self):
        with st.expander('Data Analysis', expanded=False):
            st.slider(
                label='Downsampling Size',
                min_value=1,
                max_value=500,
                step=1,
                value=100,
                key='downsampling_size',
            )
            analyze_btn = st.button("Analyze", use_container_width=True)
            if analyze_btn:
                with st.spinner('Wait for analyze...'):
                    if st.session_state.get('downsampled_data_path', None) is None:
                        downsampled_data_path = downsample(
                            data_path=st.session_state.dataset_path,
                            n=st.session_state.downsampling_size,
                        )
                        st.session_state.downsampled_data_path = downsampled_data_path
                        word_cloud_path = word_cloud(data_path=st.session_state.downsampled_data_path)
                        st.session_state.word_cloud_path = word_cloud_path
                    cfg_path = self.op_pool.export_config(
                        project_name=st.session_state.project_name,
                        dataset_path=st.session_state.downsampled_data_path,
                        nproc=4,
                        export_path="./outputs/processed_data.jsonl",
                        config_path="./configs/demo-analyze.yaml"
                    )
                    args_in_cmd = ['--config', cfg_path]
                    cfg = init_configs(args=args_in_cmd)
                    cfg['save_stats_in_one_file'] = True
                    cfg['percentiles'] = [0.01 * i for i in range(101)]
                    cfg['save_stats_in_one_file'] = True
                    logger.info('analyze data')
                    analyzer = Analyzer(cfg)
                    analyzed_dataset = analyzer.run()

                    overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
                    analysis_res_ori = pd.DataFrame()
                    if os.path.exists(overall_file):
                        analysis_res_ori = pd.read_csv(overall_file)

                    st.session_state.original_overall = analysis_res_ori
                    st.session_state.analyzed_dataset = analyzed_dataset

                    enabled_ops = [op_name for op_name in self.op_pool if self.op_pool[op_name].enabled]
                    for op_name in enabled_ops:
                        if self.op_pool[op_name].dj_stats_key is None:
                            continue
                        stats_raw = list(analysis_res_ori[self.op_pool[op_name].dj_stats_key])
                        stats = dict(
                            count=int(stats_raw[0]),
                            mean=stats_raw[1],
                            std=stats_raw[2],
                            min=stats_raw[3],
                            max=stats_raw[104],
                            quantiles=stats_raw[4:104],
                        )
                        self.op_pool[op_name].update_with_stats(stats)
                    self.op_pool.save()
                    st.rerun()
            if st.session_state.get('original_overall', None) is not None:
                display_dataset_details = st.checkbox('Display dataset details')
                if display_dataset_details:
                    analyzed_dataset = st.session_state.get(
                        'analyzed_dataset', None)
                    st.dataframe(analyzed_dataset, use_container_width=True)
                    if st.session_state.get('word_cloud_path', None) is not None:
                        st.image(st.session_state.word_cloud_path)
                display_stats = st.checkbox('Display statistics')
                if display_stats:
                    original_overall = st.session_state.original_overall
                    rows_to_display = ['mean', 'std', 'min', '5%', '10%', '25%', '50%', '75%', '90%', '95%', 'max']
                    displayed_stats = original_overall.loc[original_overall['Unnamed: 0'].isin(rows_to_display)]
                    st.dataframe(displayed_stats, use_container_width=True)
                original_imgs = st.session_state.get('original_imgs', [])
                for img in original_imgs:
                    st.image(img, output_format='png')

    def attribution(self):
        with st.expander('Data Attribution', expanded=False):
            example_valid_data_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             './data/demo-valid-dataset.jsonl'))
            st.text_area(label='Validation Data Path',
                         key='valid_data_path',
                         value=f'{example_valid_data_path}')
            display_valid_dataset_details = st.checkbox('Display validation dataset details')
            if display_valid_dataset_details:
                st.dataframe(load_dataset_as_df(st.session_state.valid_data_path), use_container_width=True)
            attribution_btn = st.button("Attribute", use_container_width=True)
            if attribution_btn:
                with st.spinner('Wait for attribution...'):
                    # TODO: formal attributors
                    attributor = TextEmbdSimilarityAttributor()
                    enabled_ops = [op_name for op_name in self.op_pool if self.op_pool[op_name].enabled]
                    enabled_op_stats_keys = []
                    for op_name in enabled_ops:
                        stats_key = self.op_pool[op_name].dj_stats_key
                        if stats_key is not None:
                            enabled_op_stats_keys.append(stats_key)
                    dataset = st.session_state.get(
                        "analyzed_dataset",
                        load_dataset(st.session_state.dataset_path, n=st.session_state.downsampling_size)
                    )
                    valid_dataset = load_dataset(st.session_state.valid_data_path)
                    attribution_result = attributor.run(dataset, valid_dataset, enabled_op_stats_keys)
                    st.session_state.attribution_result = attribution_result
            if st.session_state.get('attribution_result', None) is not None:
                st.dataframe(pd.DataFrame(st.session_state.attribution_result), use_container_width=True)
    
    @st.dialog("Data-Juicer Q&A Copilot")
    def copilot_dialog(self):

        if "copilot_chat_history" not in st.session_state:
            st.session_state.copilot_chat_history = []

        if st.button("Clear Conversation History", use_container_width=True):
            st.session_state.copilot_chat_history = []
            # st.rerun()

        chat_container = st.container(height=300)
        
        with chat_container:
            st.info(
            "This is a Copilot demo. The responses are generated by an AI model "
            "and are not guaranteed to be accurate or complete."
        )
            for message in st.session_state.copilot_chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about Data-Juicer..."):
            
            st.session_state.copilot_chat_history.append({"role": "user", "content": prompt})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    
                    q = queue.Queue()
                    thread = threading.Thread(target=call_service_in_thread, args=(list(st.session_state.copilot_chat_history), q))
                    thread.start()
                    
                    # Wait for the response
                    wait_flag = "Searching"
                    animation_frames = [wait_flag + ("." * i) for i in range(1, 4)]
                    frame_idx = 0
                    while thread.is_alive() and q.empty():
                        response_placeholder.markdown(animation_frames[frame_idx % len(animation_frames)])
                        frame_idx += 1
                        time.sleep(0.2)
                        
                    full_response = ""
                    while True:
                        chunk = q.get()
                        if chunk is None:
                            break
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
            st.session_state.copilot_chat_history.append({"role": "assistant", "content": full_response})
            
    def visualize(self):
        Visualize.setup()
        main_cols = st.columns([0.85, 0.15]) 
        with main_cols[1]: 
            if st.button("🤖 Ask AI", use_container_width=True):
                self.copilot_dialog()

        col1, col2 = st.columns(2)
        Visualize.register()
        self.llm_assistant()
        self.analyze_process()
        self.attribution()
        self.op_pool.render()


def main():
    viz = Visualize()
    viz.visualize()


if __name__ == '__main__':
    main()
