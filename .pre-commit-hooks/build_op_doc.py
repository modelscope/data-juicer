import ast
import json
import os
import re
from typing import Any, List

import translators as ts

DOC_PATH = 'docs/Operators.md'

# >>> some constant doc contents
DOC_ABSTRACT = '''
# Operator Schemas 算子提要

Operators are a collection of basic processes that assist in data modification,
cleaning, filtering, deduplication, etc. We support a wide range of data
sources and file formats, and allow for flexible extension to custom datasets.

算子 (Operator) 是协助数据修改、清理、过滤、去重等基本流程的集合。我们支持广泛的数据来源和文件格式，并支持对自定义数据集的灵活扩展。

This page offers a basic description of the operators (OPs) in Data-Juicer.
Users can refer to the
[API documentation](https://modelscope.github.io/data-juicer/) for the specific
parameters of each operator. Users can refer to and run the unit tests
(`tests/ops/...`) for [examples of operator-wise usage](../tests/ops) as well
as the effects of each operator when applied to built-in test data samples.
Besides, you can try to use agent to automatically route suitable OPs and
call them. E.g., refer to
[Agentic Filters of DJ](../demos/api_service/react_data_filter_process.ipynb),
 [Agentic Mappers of DJ](../demos/api_service/react_data_mapper_process.ipynb)

这个页面提供了OP的基本描述，用户可以参考[API文档](https://modelscope.github.io/data-juicer/)更细致了解每个
OP的具体参数，并且可以查看、运行单元测试 (`tests/ops/...`)，来体验
[各OP的用法示例](../tests/ops)以及每个OP作用于内置测试数据样本时的效果。例如，参考
[Agentic Filters of DJ](../demos/api_service/react_data_filter_process.ipynb),
 [Agentic Mappers of DJ](../demos/api_service/react_data_mapper_process.ipynb)
'''

DOC_CONTRIBUTING = '''
## Contributing  贡献

We welcome contributions of adding new operators. Please refer to [How-to Guide
for Developers](DeveloperGuide.md).

我们欢迎社区贡献新的算子，具体请参考[开发者指南](DeveloperGuide_ZH.md)。
'''

OP_TYPE_DESC = {
    'formatter':
    'Discovers, loads, and canonicalizes source data. 发现、加载、规范化原始数据。',
    'mapper':
    'Edits and transforms samples. 对数据样本进行编辑和转换。',
    'filter':
    'Filters out low-quality samples. 过滤低质量样本。',
    'deduplicator':
    'Detects and removes duplicate samples. 识别、删除重复样本。',
    'selector':
    'Selects top samples based on ranking. 基于排序选取高质量样本。',
    'grouper':
    'Group samples to batched samples. 将样本分组，每一组组成一个批量样本。',
    'aggregator':
    'Aggregate for batched samples, such as summary or conclusion. '
    '对批量样本进行汇总，如得出总结或结论。',
}
# <<<

# >>> OP code/test paths and exclusive files/dirs
OP_CODE_PREFIX = 'data_juicer/ops/'
OP_TEST_PREFIX = 'tests/ops/'
OP_EXCLUDE = {'__init__.py', 'common', '__pycache__'}

FORMATTER_CODE_PREFIX = 'data_juicer/format/'
FORMATTER_TEST_PREFIX = 'tests/format/'
FORMATTER_EXCLUDE = {'__init__.py', 'load.py'}
# <<<

# load OP tag mappings
ALL_TAG_MAPPING = json.load(
    open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     'tag_mappings.json')))


def replace_tags_with_icons(tags, lang='en'):
    icons = []
    for tag in tags:
        for tag_type in ALL_TAG_MAPPING:
            tag_mapping = ALL_TAG_MAPPING[tag_type]
            if tag in tag_mapping:
                icons.append(tag_mapping[tag]['icon'])
                break
    return icons


def remove_emojis(text):
    # This pattern includes a wide range of emoji characters
    emoji_pattern = re.compile(
        '['  # Start of character class
        '\U0001F600-\U0001F64F'  # Emoticons
        '\U0001F300-\U0001F5FF'  # Misc Symbols and Pictographs
        '\U0001F680-\U0001F6FF'  # Transport and Map Symbols
        '\U0001F700-\U0001F77F'  # Alchemical Symbols
        '\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
        '\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
        '\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        '\U0001FA00-\U0001FA6F'  # Chess Symbols
        '\U0001F000-\U0001F02F'  # Mahjong Tiles
        '\U0001F0A0-\U0001F0FF'  # Playing Cards
        '\U00002700-\U000027BF'  # Dingbats
        '\U0001F1E6-\U0001F1FF'  # Regional Indicator Symbols
        ']+',  # One or more of the above
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)  # Replace emojis with an empty string


# OP tag analysis functions
def analyze_modality_tag(code, op_prefix):
    """
    Analyze the modality tag for the given code content string. Should be one
    of the "Modality Tags" in `tagging_mappings.json`. It makes the choice by
    finding the usages of attributes `{modality}_key` and the prefix of the OP
    name. If there are multiple modality keys are used, the 'multimodal' tag
    will be returned instead.
    """
    tags = []
    if 'self.text_key' in code or op_prefix == 'text':
        tags.append('text')
    if 'self.image_key' in code or op_prefix == 'image':
        tags.append('image')
    if 'self.audio_key' in code or op_prefix == 'audio':
        tags.append('audio')
    if 'self.video_key' in code or op_prefix == 'video':
        tags.append('video')
    if len(tags) > 1:
        tags = ['multimodal']
    return tags


def analyze_resource_tag(code):
    """
    Analyze the resource tag for the given code content string. Should be one
    of the "Resource Tags" in `tagging_mappings.json`. It makes the choice
    according to their assigning statement to attribute `_accelerator`.
    """
    if '_accelerator = \'cuda\'' in code:
        return ['gpu']
    else:
        return ['cpu']


def analyze_model_tags(code):
    """
    Analyze the model tag for the given code content string. SHOULD be one of
    the "Model Tags" in `tagging_mappings.json`. It makes the choice by finding
    the `model_type` arg in `prepare_model` method invocation.
    """
    pattern = r'model_type=[\'|\"](.*?)[\'|\"]'
    groups = re.findall(pattern, code)
    tags = []
    for group in groups:
        if group == 'api':
            tags.append('api')
        elif group == 'vllm':
            tags.append('vllm')
        elif group in {
                'huggingface', 'diffusion', 'simple_aesthetics', 'video_blip'
        }:
            tags.append('hf')
    return tags


def analyze_tag_from_code(code_path):
    """
    Analyze the tags for the OP from the given code path.
    """
    tags = []
    op_prefix = code_path.split('/')[-1].split('_')[0]
    with open(code_path, 'r', encoding='utf-8') as fin:
        content = fin.read()
        # analyze modality
        tags.extend(analyze_modality_tag(content, op_prefix))
        tags.extend(analyze_resource_tag(content))
        tags.extend(analyze_model_tags(content))
    return tags


# <<<


class OPRecord:
    """
    OP record class to represent the OP record to be shown in the OP list of
    the doc.
    """

    def __init__(self,
                 type: str,
                 name: str,
                 desc: str,
                 tags: List[str] = None,
                 code: str = None,
                 test: str = None):
        self.type = type
        self.name = name
        self.tags = tags if tags else []
        self.desc = desc
        self.code = code
        self.test = test

    def __repr__(self):
        return f'{self.type}, {self.name}, {self.tags}, {self.desc}, ' \
               f'{self.code}, {self.test}'

    def __eq__(self, other):
        return self.type == other.type and self.name == other.name \
               and set(self.tags) == set(other.tags) \
               and self.desc == other.desc and self.code == other.code \
               and self.test == other.test

    def __ne__(self, other):
        return not self.__eq__(other)


class ClassVisitor(ast.NodeVisitor):
    """
    A class visitor for AST to get the doc strings of each class.
    """

    def __init__(self):
        super().__init__()
        self.docs = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        name = node.name
        node_info = ast.get_docstring(node)
        if node_info is None:
            print(f'No docstring found for class {name}')
            self.generic_visit(node)
            return
        docstring = ' '.join(node_info.split()).split('. ')[0]
        if not docstring.endswith('.'):
            docstring += '.'
        self.docs.append((name, docstring))
        self.generic_visit(node)

    def get_class_docs(self):
        return self.docs


def get_class_and_docstring(code_path):
    """
    Get the class name and its doc strings from the given Python code path.
    """
    with open(code_path, 'r', encoding='utf-8') as fin:
        code = fin.read()
        tree = ast.parse(code)
        cls_visitor = ClassVisitor()
        cls_visitor.visit(tree)
        return cls_visitor.docs


def get_op_list_from_code_for_formatter():
    """
    Get the OP record list for Formatters specifically.
    """
    op_record_list = []
    type = 'formatter'
    for formatter in os.listdir(FORMATTER_CODE_PREFIX):
        if formatter in FORMATTER_EXCLUDE:
            continue
        if formatter == 'formatter.py':
            # add record for local/remote_formatter
            code_path = os.path.join(FORMATTER_CODE_PREFIX, formatter)
            test_path = os.path.join(FORMATTER_TEST_PREFIX,
                                     'test_unify_format.py')
            docstrings = get_class_and_docstring(code_path)
            for cls, doc in docstrings:
                if cls == 'LocalFormatter':
                    name = 'local_formatter'
                elif cls == 'RemoteFormatter':
                    name = 'remote_formatter'
                else:
                    continue
                op_record_list.append(
                    OPRecord(
                        type=type,
                        name=name,
                        desc=doc,
                        code=code_path,
                        test=test_path,
                    ))
        else:
            code_path = os.path.join(FORMATTER_CODE_PREFIX, formatter)
            test_path = os.path.join(FORMATTER_TEST_PREFIX,
                                     f'test_{formatter}')
            if os.path.isdir(code_path):
                continue
            docstrings = get_class_and_docstring(code_path)
            _, doc = docstrings[0]
            op_record_list.append(
                OPRecord(
                    type=type,
                    name=formatter.replace('.py', ''),
                    desc=doc,
                    code=code_path,
                    test=test_path if os.path.exists(test_path) else '-',
                ))
    return op_record_list


def get_op_list_from_code():
    """
    Get the OP record list for regular OPs (except Formatters).
    """
    # get docs for formatters first
    op_record_list = get_op_list_from_code_for_formatter()
    # get docs for other ops
    for type in os.listdir(OP_CODE_PREFIX):
        if type in OP_EXCLUDE:
            continue
        type_dir = os.path.join(OP_CODE_PREFIX, type)
        if os.path.isfile(type_dir):
            continue
        for op in os.listdir(type_dir):
            if op in OP_EXCLUDE:
                continue
            code_path = os.path.join(type_dir, op)
            test_path = os.path.join(OP_TEST_PREFIX, type, f'test_{op}')
            if os.path.isdir(code_path):
                continue
            docstrings = get_class_and_docstring(code_path)
            _, doc = docstrings[0]
            op_record_list.append(
                OPRecord(
                    type=type,
                    name=op.replace('.py', ''),
                    desc=doc,
                    tags=analyze_tag_from_code(code_path),
                    code=code_path,
                    test=test_path if os.path.exists(test_path) else '-',
                ))
    op_record_list.sort(key=lambda record: (record.type, record.name))
    return op_record_list


def generate_new_doc(op_record_list):
    """
    Generate new docs for the updated OP records.
    """
    op_record_dict = {}
    for record in op_record_list:
        op_record_dict.setdefault(record.type, []).append(record)
    # initialize with abstraction
    doc = [DOC_ABSTRACT]
    # make overview
    doc.append(generate_overview(op_record_dict))
    # make OP tables
    for op_type, op_records in op_record_dict.items():
        doc.append(generate_op_table_section(op_type, op_records))
    # add
    doc.append(DOC_CONTRIBUTING)

    # write to doc file
    output_doc_path = DOC_PATH
    with open(output_doc_path, 'w', encoding='utf-8') as fout:
        fout.write('\n\n'.join(doc))


def generate_overview(op_record_dict):
    """
    Generate the overview section according to the OP record dict categorized
    by their types.
    """
    # make the header
    doc = ['## Overview  概览']
    # make the summarization.
    doc.append(f'The operators in Data-Juicer are categorized into '
               f'{len(op_record_dict)} types.\nData-Juicer 中的算子分为以下 '
               f'{len(op_record_dict)} 种类型。')
    # make the type table.
    table = [
        '| Type 类型 | Number 数量 | Description 描述 |',
        '|------|:------:|-------------|',
    ]
    for type in op_record_dict:
        table.append(f'| [{type}](#{type}) | {len(op_record_dict[type])} | '
                     f'{OP_TYPE_DESC[type]} |')
    doc.append('\n'.join(table))
    # make tag description
    tag_intro = [
        'All the specific operators are listed below, each featured with '
        'several capability tags. \n下面列出所有具体算子，每种算子都通过多个标签来注明其主要功能。'
    ]
    for tag_type in ALL_TAG_MAPPING:
        tag_intro.append(f'* {tag_type}')
        tag_mapping = ALL_TAG_MAPPING[tag_type]
        for tag in tag_mapping:
            tag_icon = tag_mapping[tag]['icon']
            tag_desc = tag_mapping[tag]['desc']
            tag_intro.append(f'  - {tag_icon}: {tag_desc}')
    doc.append('\n'.join(tag_intro))
    return '\n\n'.join(doc)


def generate_op_table_section(op_type, op_record_list):
    """
    Generate the OP table section for the given OP type and the OP record list.
    """
    # make the header
    doc = [f'## {op_type} <a name="{op_type}"/>']
    # make the OP table
    table = [
        '| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 |'
        ' Unit tests 单测样例 |',
        '|----------|------|-------------|-------------|------------|'
    ]
    trans_descs = get_op_desc_in_en_zh_batched(
        [record.desc for record in op_record_list])
    for i, record in enumerate(op_record_list):
        tags = ' '.join(replace_tags_with_icons(record.tags))
        tests = f'[tests]({os.path.join("..", record.test)})' \
            if record.test != '-' else '-'
        op_row = f'| {record.name} ' \
                 f'| {tags} ' \
                 f'| {trans_descs[i]} ' \
                 f'| [code]({os.path.join("..", record.code)}) ' \
                 f'| {tests} |'
        table.append(op_row)
    doc.append('\n'.join(table))
    return '\n\n'.join(doc)


def get_op_desc_in_en_zh_batched(descs):
    separator = '\n'
    batch = separator.join(descs)
    res = ts.translate_text(batch,
                            translator='alibaba',
                            from_language='en',
                            to_language='zh')
    zhs = res.split(separator)
    assert len(zhs) == len(descs)
    return [desc + ' ' + zh.strip() for desc, zh in zip(descs, zhs)]


def parse_op_record_from_current_doc():
    """
    Parse the old-version OP records from the existing OP doc.
    """
    # patterns
    tab_pattern = r'\| +(.*?) +\| +(.*?) +\| +(.*?) +\| +(.*?) +\| +(.*?) +\|'
    link_pattern = r'\[.*?\]\((.*?)\)'

    if os.path.exists(DOC_PATH):
        op_record_list = []
        with open(DOC_PATH, 'r', encoding='utf-8') as fin:
            content = fin.read()
            res = re.findall(tab_pattern, content)
            for name, tags, desc, code, test in res:
                # skip table header
                if name == 'Operator 算子':
                    continue
                # extract tags
                type = name.split('_')[-1]
                tags = [remove_emojis(tag.lower()) for tag in tags.split(' ')]
                # only need English description
                desc = desc.split('. ')[0] + '.'
                code = re.findall(link_pattern, code)[0]
                test = re.findall(link_pattern, test)
                op_record_list.append(
                    OPRecord(type=type,
                             name=name,
                             desc=desc,
                             tags=tags,
                             code=code.replace('../', ''),
                             test=test[0].replace('../', '')
                             if len(test) > 0 else '-'))
        op_record_list.sort(key=lambda record: (record.type, record.name))
        return op_record_list
    else:
        return []


def check_and_update_op_record(old_op_record_list, new_op_record_list):
    """
    Update states in the new OP records based on the old version.

    The update categories cover:
    1. usability tags update
        1.1 If there is no unittest for this OP, set it to alpha;
            otherwise, set it to beta.
        1.2 Then if it's beta in the new version, but it's *mannally* checked
            and set to be stable in the old version,
            the final tag will be overrided as stable.

    | old tag | new tag | res tag |
    |---|---|---|
    | alpha | alpha | alpha |
    | alpha | beta | beta |
    | beta | alpha | alpha |
    | beta | beta | beta |
    | stable | alpha | alpha |
    | stable | beta | **stable** |
    """
    usability_tag_set = set(ALL_TAG_MAPPING['Usability Tags'].keys())
    old_op_record_dict = {record.name: record for record in old_op_record_list}
    updated_op_record_list = []
    for record in new_op_record_list:
        # check unittest
        test = record.test
        if not test or test == '-' or not os.path.exists(test):
            usability_tag = 'alpha'
        else:
            usability_tag = 'beta'
        if record.name in old_op_record_dict:
            # get the old usability tag
            old_record = old_op_record_dict[record.name]
            old_usability_tag = None
            for tag in old_record.tags:
                if tag in usability_tag_set:
                    old_usability_tag = tag
                    break
            if old_usability_tag and \
                    old_usability_tag == 'stable' and usability_tag == 'beta':
                print(f'{record.name} kept stable')
                usability_tag = 'stable'
        curr_tags = [
            tag for tag in record.tags if tag not in usability_tag_set
        ]
        curr_tags.append(usability_tag)
        record.tags = curr_tags
        updated_op_record_list.append(record)

    return updated_op_record_list


def main():
    old_op_record_list = parse_op_record_from_current_doc()
    new_op_record_list = get_op_list_from_code()
    updated_op_record_list = check_and_update_op_record(
        old_op_record_list, new_op_record_list)
    # if the doc is changed, exit with non-zero value
    if old_op_record_list == updated_op_record_list:
        exit(0)
    else:
        generate_new_doc(updated_op_record_list)
        print('Operator document is updated.')
        exit(1)


if __name__ == '__main__':
    main()
