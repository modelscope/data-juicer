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

op_types = ['mapper', 'filter',]# 'deduplicator'] , 'selector']
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
    return ''.join(text[0])

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

def create_mapper_tab(op_type, op_tab):
    with op_tab:
        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")
        gr.Markdown(" **Op Parameters**")
        op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
        with gr.Column():
            with gr.Group('Inputs'):
                gr.Markdown(" **Inputs**")
                with gr.Row():
                    # img = '/private/var/folders/7b/p5l9gykj1k7_tylkvwjv_sl00000gp/T/gradio/f24972121fd4d4f95f42f1cd70f859bb03839e76/image_blur_mapper/喜欢的书__dj_hash_#14a7b2e1b96410fbe63ea16a70422180db53d644661630938b2773d8efa18dde#.png'
                    
                    input_text = gr.TextArea(label="Text",interactive=True,)
                    input_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    input_video = gr.Video(label='Video', visible=multimodal)
                    input_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)
            with gr.Group('Outputs'):
                gr.Markdown(" **Outputs**")
                with gr.Row():
                    output_text = gr.TextArea(label="Text",interactive=False,)
                    output_image = gr.Image(label='Image', visible=multimodal)
                    output_video = gr.Video(label='Video', visible=multimodal)
                    output_audio = gr.Audio(label='Audio', visible=multimodal)
            code = gr.Code(label='Source', language='python')
        def run_op(op_name, op_params, input_text, input_image, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = dict()
            
            sample[text_key] = input_text
            sample[image_key] = [input_image]
            sample[video_key] = [input_video]
            sample[audio_key] = [input_audio]
            
            output_sample = op.process(copy.deepcopy(sample))
            
            return decode_sample(output_sample)
        
        inputs = [op_selector, op_params, input_text, input_image, input_video, input_audio]
        outputs = [output_text, output_image, output_video, output_audio]
        run_button.click(run_op, inputs=inputs, outputs=outputs)
        show_code_button.click(show_code, inputs=[op_selector], outputs=[code])

def create_filter_tab(op_type, op_tab):
    with op_tab:

        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")
        gr.Markdown(" **Op Parameters**")
        op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
        with gr.Column():
            with gr.Group('Inputs'):
                gr.Markdown(" **Inputs**")
                with gr.Row():
                    input_text = gr.TextArea(label="Text",interactive=True,)
                    input_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    input_video = gr.Video(label='Video', visible=multimodal)
                    input_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)
                    input_stats = gr.Json(label='Stats')

            with gr.Group('Outputs'):
                gr.Markdown(" **Outputs**")
                with gr.Row():
                    output_text = gr.TextArea(label="Text",interactive=False,)
                    output_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    output_video = gr.Video(label='Video', visible=multimodal)
                    output_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)
                    output_stats = gr.Json(label='Stats')

            code = gr.Code(label='Source', language='python')
        def run_op(op_name, op_params, input_text, input_image, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = dict()
            sample[Fields.stats] = dict()
            sample[text_key] = input_text
            sample[image_key] = [input_image]
            sample[video_key] = [input_video]
            sample[audio_key] = [input_audio]
            input_stats = sample[Fields.stats]
            output_sample = op.compute_stats(copy.deepcopy(sample))
            output_stats = output_sample[Fields.stats]   
            return *decode_sample(output_sample), input_stats, output_stats
        
        inputs = [op_selector, op_params, input_text, input_image, input_video, input_audio]
        outputs = [output_text, output_image, output_video, output_audio, input_stats, output_stats]
        run_button.click(run_op, inputs=inputs, outputs=outputs)
        show_code_button.click(show_code, inputs=[op_selector], outputs=[code])

def create_deduplicator_tab(op_type, op_tab):
    with op_tab:
        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")
        gr.Markdown(" **Op Parameters**")
        op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
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

            code = gr.Code(label='Source', language='python')
        def run_op(op_name, op_params, input_text, input_images, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = dict()
            sample[text_key] = input_text
            sample[image_key] = input_images
            sample[video_key] = [input_video]
            sample[audio_key] = [input_audio]
            
            output_sample = sample #op.compute_hash(copy.deepcopy(sample))   
            return decode_sample(output_sample)
        
        inputs = [op_selector, op_params, input_text, input_image, input_video, input_audio]
        outputs = [output_text, output_image, output_video, output_audio]
        run_button.click(run_op, inputs=inputs, outputs=outputs)
        show_code_button.click(show_code, inputs=[op_selector], outputs=[code])

def create_selector_tab(op_type, op_tab):
    with op_tab:
        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")
        gr.Markdown(" **Op Parameters**")
        op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
        with gr.Column():
            with gr.Group('Inputs'):
                gr.Markdown(" **Inputs**")
                with gr.Row():
                    input_text = gr.TextArea(label="Text",interactive=True,)
                    input_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    input_video = gr.Video(label='Video', visible=multimodal)
                    input_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)
                    input_stats = gr.Json(label='Stats')

            with gr.Group('Outputs'):
                gr.Markdown(" **Outputs**")
                with gr.Row():
                    output_text = gr.TextArea(label="Text",interactive=False,)
                    output_image = gr.Image(label='Image', type='filepath', visible=multimodal)
                    output_video = gr.Video(label='Video', visible=multimodal)
                    output_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal)
                    output_stats = gr.Json(label='Stats')

            code = gr.Code(label='Source', language='python')
        def run_op(op_name, op_params, input_text, input_image, input_video, input_audio):
            op_class = OPERATORS.modules[op_name]
            try:
                params = yaml.safe_load(op_params)
            except:
                params = {}
            if params is None:
                params = {}
            op = op_class(**params)
            sample = dict()
            sample[Fields.stats] = dict()
            sample[text_key] = input_text
            sample[image_key] = [input_image]
            sample[video_key] = [input_video]
            sample[audio_key] = [input_audio]
            input_stats = sample[Fields.stats]
            output_sample = op.compute_stats(copy.deepcopy(sample))
            output_stats = output_sample[Fields.stats]
   
            return *decode_sample(output_sample), input_stats, output_stats
        
        inputs = [op_selector, op_params, input_text, input_image, input_video, input_audio]
        outputs = [output_text, output_image, output_video, output_audio, input_stats, output_stats]
        run_button.click(run_op, inputs=inputs, outputs=outputs)
        show_code_button.click(show_code, inputs=[op_selector], outputs=[code])

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
