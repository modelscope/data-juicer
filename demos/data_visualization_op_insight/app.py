import os
import inspect
import base64
import yaml
import copy
import shutil
import gradio as gr
from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields
demo_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(demo_path))


# 图片本地路径转换为 base64 格式
def covert_image_to_base64(image_path):
    # 获得文件后缀名
    ext = image_path.split(".")[-1]
    if ext not in ["gif", "jpeg", "png"]:
        ext = "jpeg"

    with open(image_path, "rb") as image_file:
        # Read the file
        encoded_string = base64.b64encode(image_file.read())

        # Convert bytes to string
        base64_data = encoded_string.decode("utf-8")

        # 生成base64编码的地址
        base64_url = f"data:image/{ext};base64,{base64_data}"
        return base64_url


def format_cover_html(project_img_path):
    readme_link = 'https://github.com/alibaba/data-juicer'
    config = {
        'name': "Data-Juicer",
        'label': "Op Insight",
        'description': f'A One-Stop Data Processing System for Large Language Models, see more details in <a href="{readme_link}">GitHub</a>',
        'introduction': 
        "This project is being actively updated and maintained, and we will periodically enhance and add more features and data recipes. <br>"
        "We welcome you to join us in promoting LLM data development and research!<br>",
        'demo':"You can experience the effect of the operators of Data-Juicer"
    }
    # image_src = covert_image_to_base64(project_img_path)
    # <div class="project_img"> <img src={image_src} /> </div>
    # <div class='project_cover'>
    return f"""
    <div>
    <div class="project_name">{config.get("name", "")} </div>
    <div class="project_desc">{config.get("description", "")}</div>
    <div class="project_desc">{config.get("introduction", "")}</div>
    <div class="project_desc">{config.get("demo", "")}</div>
</div>
"""
op_text = ''
docs_file = os.path.join(project_path, 'docs/Operators.md')
if os.path.exists(docs_file):
    with open(os.path.join(project_path, 'docs/Operators.md'), 'r') as f:
        op_text = f.read()

def extract_op_desc(markdown_text,  header):
    start_index = markdown_text.find(header)
    end_index = markdown_text.find("\n##", start_index + len(header)) 
    return markdown_text[start_index+ len(header):end_index].strip()

op_desc = f"<div style='text-align: center;'>{extract_op_desc(op_text, '## Overview').split('All the specific ')[0].strip()}</div>"
op_list_desc = {
    'mapper':extract_op_desc(op_text, '## Mapper <a name="mapper"/>'),
    'filter':extract_op_desc(op_text, '## Filter <a name="filter"/>'),
    'deduplicator':extract_op_desc(op_text, '## Deduplicator <a name="deduplicator"/>'),
    'selector':extract_op_desc(op_text, '## Selector <a name="selector"/>'),
}

op_types = ['filter',]# 'deduplicator'] , 'selector']
local_ops_dict = {op_type:[] for op_type in op_types}
multimodal = os.getenv('MULTI_MODAL', False)
multimodal = True
text_key = 'text'
image_key = 'images'
audio_key = 'audios'
video_key = 'videos'

def get_op_lists(op_type):
    use_local_op = os.getenv('USE_LOCAL_OP', False)
    if not use_local_op:
        all_ops = list(OPERATORS.modules.keys())
        options = [
            name for name in all_ops if name.endswith(op_type)
        ]
    else:
        options = local_ops_dict.get(op_type, [])

    for exclude in ['image', 'video', 'audio']:
        options = [name for name in options if multimodal or exclude not in name]
    return options

def show_code(op_name):
    op_class = OPERATORS.modules[op_name]
    text = inspect.getsourcelines(op_class)

    init_signature = inspect.signature(op_class.__init__)

    # 输出每个参数的名字和默认值
    default_params = dict()
    for name, parameter in init_signature.parameters.items():
        if name in ['self', 'args', 'kwargs']:
            continue  # 跳过 'self' 参数
        if parameter.default is not inspect.Parameter.empty:
            default_params[name] = parameter.default

    return ''.join(text[0]), yaml.dump(default_params)

def encode_sample(input_text, input_image, input_video, input_audio):
    sample = dict()
    sample[text_key]=input_text
    sample[image_key]= input_image #[input_image] if input_image else []
    sample[video_key]=[input_video] if input_video else []
    sample[audio_key]=[input_audio] if input_audio else []
    return sample

def decode_sample(output_sample):
    output_text = output_sample[text_key]
    output_image = output_sample[image_key][0] if output_sample[image_key] else None
    output_video = output_sample[video_key][0] if output_sample[video_key] else None 
    output_audio = output_sample[audio_key][0] if output_sample[audio_key] else None
    def copy_func(file):
        filename = None
        if file:
            filename= os.path.basename(file)
            shutil.copyfile(file, filename)
        return filename
    
    image_file = copy_func(output_image)
    video_file = copy_func(output_video)
    audio_file = copy_func(output_audio)
    return output_text, image_file, video_file, audio_file

def create_tab_layout(op_tab, op_type, run_op, has_stats=False):
    with op_tab:
        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            with gr.Column():
                gr.Markdown(" **Op Parameters**")
                op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")

        with gr.Column():
            with gr.Group('Inputs'):
                gr.Markdown(" **Inputs**")
                with gr.Row():
                    input_text = gr.TextArea(label="Text",interactive=True,)
                    input_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    input_video = gr.Video(label='Video', visible=multimodal)
                    input_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)

            with gr.Group('Outputs'):
                gr.Markdown(" **Outputs**")
                with gr.Row():
                    output_text = gr.TextArea(label="Text",interactive=False,)
                    output_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    output_video = gr.Video(label='Video', visible=multimodal)
                    output_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)
                    if has_stats:
                        with gr.Column():
                            output_stats = gr.Json(label='Stats')
                            output_keep = gr.Text(label='Keep or not?', interactive=False)

            code = gr.Code(label='Source', language='python')
        inputs = [op_selector, op_params, input_text, input_image, input_video, input_audio]
        outputs = [output_text, output_image, output_video, output_audio]
        if has_stats:
            outputs.append(output_stats)
            outputs.append(output_keep)

        def run_func(*args):
            try:
                return run_op(*args)
            except Exception as e:
                gr.Error(str(e))
                return outputs

        def on_tab_select():
            return gr.Dropdown(value=options[-1], label=label, choices=options, interactive=True)
        show_code_button.click(show_code, inputs=[op_selector], outputs=[code, op_params])
        run_button.click(run_func, inputs=inputs, outputs=outputs)
        op_selector.change(show_code, inputs=[op_selector], outputs=[code, op_params])
        # op_selector.select(show_code, inputs=[op_selector], outputs=[code, op_params])
    op_tab.select(on_tab_select, outputs=[op_selector])
def create_mapper_tab(op_type, op_tab):
    with op_tab:
        def run_op(op_name, op_params, input_text, input_image, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = encode_sample(input_text, input_image, input_video, input_audio)
            output_sample = op.process(copy.deepcopy(sample))
            return decode_sample(output_sample)
        
        create_tab_layout(op_tab, op_type, run_op)


def create_filter_tab(op_type, op_tab):

    def run_op(op_name, op_params, input_text, input_image, input_video, input_audio):
        op_class = OPERATORS.modules[op_name]
        try:
            params = yaml.safe_load(op_params)
        except:
            params = {}
        if params is None:
            params = {}
        op = op_class(**params)
        sample = encode_sample(input_text, input_image, input_video, input_audio)
        sample[Fields.stats] = dict()
        output_sample = op.compute_stats(copy.deepcopy(sample))
        if op.process(output_sample):
            output_keep = 'Yes'
        else:
            output_keep = 'No'
        output_stats = output_sample[Fields.stats]
        return *decode_sample(output_sample), output_stats, output_keep
    create_tab_layout(op_tab, op_type, run_op, has_stats=True)

def create_deduplicator_tab(op_type, op_tab):
    with op_tab:
        def run_op(op_name, op_params, input_text, input_images, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = encode_sample(input_text, input_image, input_video, input_audio)
            output_sample = sample #op.compute_hash(copy.deepcopy(sample))   
            return decode_sample(output_sample)
        create_tab_layout(op_tab, op_type, run_op, has_stats=True)

def create_selector_tab(op_type, op_tab):
    with op_tab:
        def run_op(op_name, op_params, input_text, input_image, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = encode_sample(input_text, input_image, input_video, input_audio)
            sample[Fields.stats] = dict()
            input_stats = sample[Fields.stats]
            output_sample = op.compute_stats(copy.deepcopy(sample))
            output_stats = output_sample[Fields.stats]
            return *decode_sample(output_sample), input_stats, output_stats
        create_tab_layout(op_tab, op_type, run_op, has_stats=True)

with gr.Blocks(css="./app.css") as demo:
    
    dj_image = os.path.join(project_path, 'docs/imgs/data-juicer.jpg')
    gr.HTML(format_cover_html(dj_image))
    
    with gr.Accordion(label='Op Insight',open=True):
        tabs = gr.Tabs()
        with tabs:
            op_tabs = {op_type: gr.Tab(label=op_type.capitalize() + 's') for op_type in op_types}
            for op_type, op_tab in op_tabs.items():
                    create_op_tab_func = globals().get(f'create_{op_type}_tab', None)
                    if callable(create_op_tab_func):       
                        create_op_tab_func(op_type, op_tab)
                    else:
                        gr.Error(f'{op_type} not callable')

    demo.launch()
