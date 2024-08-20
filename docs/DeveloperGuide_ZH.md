# 开发者指南

- [开发者指南](#开发者指南)
  - [编码规范](#编码规范)
  - [构建自己的算子](#构建自己的算子)
    - [（可选）使新算子可以进行算子融合](#可选使新算子可以进行算子融合)
  - [构建自己的配置](#构建自己的配置)
    - [丰富的配置源和类型提示](#丰富的配置源和类型提示)
    - [层次化的配置和帮助](#层次化的配置和帮助)

## 编码规范

我们将编码规范定义在 `.pre-commit-config.yaml` 中。在向仓库贡献代码之前，请使用 `pre-commit` 工具对代码进行自动规范化。

```shell
# ===========install pre-commit tool===========
pip install pre-commit

cd <path_to_data_juicer>
# install pre-commit script for data_juicer
pre-commit install


# ===========check all files===========
git add .
pre-commit run --all-files

# commit after all checking are passed
git commit -m "<your_commit_message>"
```

**注意**：我们在github workflow配置了pre-commit的检查。如果您的PR中该检查没通过，请在本地①确保pre-commit 的相关依赖与项目配置一致（可通过`pre-commit clean`和`pre-commit install`完成）；②push前执行了`pre-commit run --all-files`.

## 构建自己的算子

- Data-Juicer 支持每个人定义自己的算子。
- 在实现新的算子之前，请参考 [Operators](Operators_ZH.md) 以避免不必要的重复。
- 假设要添加一个名为 “TextLengthFilter” 的运算符以过滤仅包含预期文本长度的样本语料，可以按照以下步骤进行构建。

1. (可选) 在 `data_juicer/utils/constant.py` 文件中添加一个新的StatsKeys来保存新算子的统计变量。

```python
class StatsKeys(object):
    ...              # other keys
    text_len = 'text_len'
```

2. 在 `data_juicer/ops/filter/` 目录下创建一个新的算子文件 `text_length_filter.py`，内容如下：
    - 因为它是一个 Filter 算子，所以需要继承 `base_op.py` 中的 `Filter` 基类，并用 `OPERATORS` 修饰以实现自动注册。

    ```python
    import sys

    from jsonargparse.typing import PositiveInt

    from data_juicer.utils.constant import Fields, StatsKeys

    from ..base_op import OPERATORS, Filter


    @OPERATORS.register_module('text_length_filter')
    class TextLengthFilter(Filter):
        """Filter to keep samples with total text length within a specific
        range."""

        def __init__(self,
                    min_len: PositiveInt = 10,
                    max_len: PositiveInt = sys.maxsize,
                    *args,
                    **kwargs):
            """
            Initialization method.

            :param min_len: The min text length in the filtering. samples
                will be filtered if their text length is below this
                parameter.
            :param max_len: The max text length in the filtering. samples
                will be filtered if their text length exceeds this
                parameter.
            :param args: extra args
            :param kwargs: extra args
            """
            super().__init__(*args, **kwargs)
            self.min_len = min_len
            self.max_len = max_len

        def compute_stats(self, sample):
            # check if it's computed already
            if StatsKeys.text_len in sample[Fields.stats]:
                return sample

            sample[Fields.stats][StatsKeys.text_len] = len(sample[self.text_key])
            return sample

        def process(self, sample):
            if self.min_len <= sample[Fields.stats][StatsKeys.text_len] <= self.max_len:
                return True
            else:
                return False
    ```

    - 如果在算子中使用了 Hugging Face 模型，您可能希望利用 GPU 加速。为了实现这一点，请在构造函数中声明 `_accelerator = 'cuda'`，并确保 `compute_stats` 和 `process` 方法接受一个额外的位置参数 `rank`。

    ```python
    # ... (same as above)

    @OPERATORS.register_module('text_length_filter')
    class TextLengthFilter(Filter):
   
        _accelerator = 'cuda'
   
        def __init__(self,
                    min_len: PositiveInt = 10,
                    max_len: PositiveInt = sys.maxsize,
                    *args,
                    **kwargs):
            # ... (same as above)

        def compute_stats(self, sample, rank=None):
            # ... (same as above)

        def process(self, sample, rank=None):
            # ... (same as above)
    ```

    - 如果算子批量处理数据，输入不是一个样本而是一个batch，需要声明`_batched_op = True`。
    ```python
    # ... (import some other libraries)
    OP_NAME = 'image_diffusion_mapper'
    @OPERATORS.register_module(OP_NAME)
    @LOADED_IMAGES.register_module(OP_NAME)
    class ImageDiffusionMapper(Mapper):
        _batched_op = True

        def __init__(self,
                 # ... (OP parameters)
                 *args,
                 **kwargs):
            super().__init__(*args, **kwargs)

        def process(self, samples):
            # ... (some codes)
    ```

    - 在mapper算子中，我们提供了产生额外数据的存储路径生成接口，避免出现进程冲突和数据覆盖的情况。生成的存储路径格式为`{ORIGINAL_DATAPATH}/__dj__produced_data__/{OP_NAME}/{ORIGINAL_FILENAME}__dj_hash_#{HASH_VALUE}#.{EXT}`，其中`HASH_VALUE`是算子初始化参数、每个样本中相关参数、进程ID和时间戳的哈希值。为了方便，可以在OP类初始化开头调用`self.remove_extra_parameters(locals())`获取算子初始化参数，同时可以调用`self.add_parameters`添加每个样本与生成额外数据相关的参数。例如，利用diffusion模型对图像进行增强的算子：
    ```python
    # ... (import some library)
    OP_NAME = 'image_diffusion_mapper'
    @OPERATORS.register_module(OP_NAME)
    @LOADED_IMAGES.register_module(OP_NAME)
    class ImageDiffusionMapper(Mapper):
        def __init__(self,
                 # ... (OP parameters)
                 *args,
                 **kwargs):
            super().__init__(*args, **kwargs)
            self._init_parameters = self.remove_extra_parameters(locals())

        def process(self, sample):
            # ... (some codes)
            # captions[index] is the prompt for diffusion model
            related_parameters = self.add_parameters(
                    self._init_parameters, caption=captions[index])
            new_image_path = transfer_filename(
                    origin_image_path, OP_NAME, **related_parameters)
            # ... (some codes)
    ```
    针对一个数据源衍生出多个额外数据的情况，我们允许在生成的存储路径后面再加后缀。比如，根据关键帧将视频拆分成多个视频：
    ```python
    # ... (import some library)
    OP_NAME = 'video_split_by_key_frame_mapper'
    @OPERATORS.register_module(OP_NAME)
    @LOADED_VIDEOS.register_module(OP_NAME)
    class VideoSplitByKeyFrameMapper(Mapper):
        def __init__(self,
                 # ... (OP parameters)
                 *args,
                 **kwargs):
            super().__init__(*args, **kwargs)
            self._init_parameters = self.remove_extra_parameters(locals())

        def process(self, sample):
            # ... (some codes)
            split_video_path = transfer_filename(
                        original_video_path, OP_NAME, **self._init_parameters)
            split_video_path = add_suffix_to_filename(split_video_path, f'_{count}')
            # ... (some codes)
    ```

3. 实现后，将其添加到 `data_juicer/ops/filter` 目录下 `__init__.py` 文件中的算子字典中：

```python
from . import (...,              # other OPs
               text_length_filter)  # import this new OP module
# other OPs
from text_length_filter import TextLengthFilter  # import this new OP class
__all__ = [
    # other Ops
    text_length_filter,  # add this new Op to __all__
]
```

4. 全部完成！现在您可以在自己的配置文件中使用新添加的算子：

```yaml
# other configs
...

# process configs
process:
  - text_length_filter:  # add this op to your process list and set the parameters
      min_len: 10
      max_len: 1000
```

5. （强烈推荐）最好为新添加的算子进行单元测试。对于上面的 `TextLengthFilter` 算子，建议在 `tests/ops/filter/` 中实现如 `test_text_length_filter.py` 的测试文件：

```python
import unittest
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextLengthFilterTest(DataJuicerTestCaseBase):

    def test_func1(self):
        pass

    def test_func2(self):
        pass

    def test_func3(self):
        pass
        
if __name__ == '__main__':
    unittest.main()
```

6. （强烈推荐）为了方便其他用户使用，我们还需要将新增的算子信息更新到相应的文档中，具体包括如下文档：
   1. `configs/config_all.yaml`：该全集配置文件保存了所有算子及参数的一个列表，作为用户参考可用算子的一个重要文档。因此，在新增算子后，需要将其添加到该文档process列表里（按算子类型分组并按字母序排序）：
   
   ```yaml
   ...
   - stopwords_filter:                                       # filter text with stopword ratio smaller than a specific min value
       lang: en                                                # consider stopwords in what language
       tokenization: false                                     # whether to use model to tokenize documents
       min_ratio: 0.3                                          # the min ratio to filter text
       stopwords_dir: ./assets                                 # directory to store stopwords dictionaries
       use_words_aug: false                                    # whether to augment words, especially for Chinese and Vietnamese
       words_aug_group_sizes: [2]                              # the group size of words to augment
       words_aug_join_char: ""                                 # the join char between words to augment
   - text_length_filter:                                     # filter text with length out of specific range
       min_len: 10                                             # the min length of filter range
       max_len: 10000                                          # the max length of filter range
   - token_num_filter:                                       # filter text with total token number out of specific range
       hf_tokenizer: EleutherAI/pythia-6.9b-deduped            # name of used Hugging Face tokenizer
       min_num: 10                                             # the min number of filter range
       max_num: 10000                                          # the max number of filter range
   ...
   ```
   
   2. `docs/Operators.md`：该文档维护了可用算子的分类列表。我们可以把新增算子的信息添加到对应类别算子的列表中（算子按字母排序）。同时，在文档最上方Overview章节，我们也需要更新对应类别的可用算子数目：
   
   ```markdown
   ## Overview
   ...
   | [ Filter ]( #filter )             |   21 (+1 HERE)   | Filters out low-quality samples                 |
   ...
   ## Filter <a name="filter"/>
   ...
   | suffix_filter                  | General | en, zh | Keeps samples with specified suffixes                                                      |
   | text_length_filter             | General | en, zh | Keeps samples with total text length within the specified range                            |
   | token_num_filter               | General | en, zh | Keeps samples with token count within the specified range                                  |
   ...
   ```

   3. `docs/Operators_ZH.md`：该文档为6.ii中`docs/Operators.md`文档的中文版，需要更新相同位置处的中文内容。


### （可选）使新算子可以进行算子融合

- 如果我们的新算子中的部分中间变量的计算过程与已有的算子重复，那么可以将其添加到可融合算子中，以在数据处理时利用算子融合进行加速。（如`words_num_filter`与`word_repetition_filter`都需要对输入文本进行分词）
- 当算子融合（OP Fusion）功能开启时，这些重复的计算过程和中间变量是可以在算子之间的`context`中共享的，从而可以减少重复计算。
- 可通过如下步骤使包含共有中间变量的算子可进行算子融合（以`words_num_filter`算子为例）。

1. （可选）如果新算子中产生了新的中间变量，需要在`utils/constant.py`中的`InterVars`类中添加新的中间变量名称。通常需要在名称前加上`DEFAULT_PREFIX`前缀。

```python
class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'  # 在这里添加新的中间变量
    ...
```

2. （可选）第1步中添加的新的中间变量还需在`ops/op_fusion.py`中为其定义一个注册组，并添加到保存了所有注册组的列表中，方便算子融合模块追踪涉及到这些中间变量的算子。

```python
...
# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)  # 为新的中间变量定义注册组

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# all
ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_IMAGES]  # 并添加到注册组列表中
...
```

3. 在涉及到该中间变量的算子前，将该算子注册到中间变量对应的注册组中，表示该算子中可能对该中间变量进行了计算与使用。

```python
...
@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)  # 将该算子注册到注册组中
class WordsNumFilter(Filter):
...
```

4. 在算子计算该中间变量的过程中，可将计算逻辑修改为：
   1. 如果`context`参数为True，则表示已开启了算子融合，优先从`context`中获取前序算子已经计算过的该中间变量的值
   2. 如果中间变量在`context`中不存在，则表示在该算子中首次对该中间变量进行计算，在计算完成后，定义一个唯一的key并将其存放到`context`中，以供后续算子使用
   3. 如果`context`参数为False，则按照正常计算流程进行

```python
# 修改计算逻辑前
...
tokenizer = get_model(self.model_key)
words = get_words_from_document(
    sample[self.text_key],
    token_func=tokenizer.encode_as_pieces if tokenizer else None)
...        

# 修改计算逻辑后
...
words_key = f'{InterVars.words}-{self.model_key}'
if context and words_key in sample[Fields.context]:
    # 直接使用context中已有的中间变量值
    words = sample[Fields.context][words_key]
else:
    # 正常计算流程
    tokenizer = get_model(self.model_key)
    words = get_words_from_document(
        sample[self.text_key],
        token_func=tokenizer.encode_as_pieces if tokenizer else None)
    if context:
        # 第一次计算该中间变量后，放入context供后续算子使用
        sample[Fields.context][words_key] = words
...
```

- 至此，该算子已经能够在算子融合功能开启后，自动地与其他算子进行融合并共享共有的中间变量，减少重复计算，加快整体的数据处理速度

## 构建自己的配置

- 我们提供基于 [jsonargparse](https://github.com/omni-us/jsonargparse/) 的简单配置以降低样板代码的成本。

### 丰富的配置源和类型提示

- 全局配置对象可以通过以下方式初始化

```python
# core.executor.py
self.cfg = init_configs()
```

- 其中可以指定和混合来自不同来源的函数参数，包括
    1. *硬编码默认值* 将配置注册到解析器中或在类的 `__init__` 函数中指定
    2. json 格式的默认*配置文件*（yaml 或 jsonnet 超集）
    3. *环境变量*
    4. *POSIX-style 命令行参数*， 例如 `--project_name my_data_demo` 或 `--project_name=my_data_demo`，包含配置文件

- 最终解析的值是来自这些来源的混合。 并且覆盖顺序与上面的数字相同。

此外，还支持许多参数类型和相应的验证。
包含 Python内置类型、来自 [Lib/typing](https://docs.python.org/3/library/typing.html) 的类型，以及来自 jsonargparse 的 [扩展类型](https://jsonargparse.readthedocs.io/en/stable/#type-hints)，例如具有自定义限制的 `restricted types` 和 `Paths`。

### 层次化的配置和帮助

- 您可以在参数名称中自由使用点符号来定义层次结构， 例如 `maximum_line_length_filter.min`.
更重要的是，默认情况下，我们自动注册已实现的运算符的 docstring。 也就是说，所有的结构配置始终与代码同步。
- 您可以通过运行脚本来获取层次化的帮助信息，例如：

```
$ python tools/process_data.py --help

usage: process_data.py [-h] [--config CONFIG] [--print_config[=flags]] [--project_name PROJECT_NAME] [--dataset_path DATASET_PATH] [--dataset_dir DATASET_DIR] [--export_path EXPORT_PATH] [--process PROCESS]
                            [--np NP] [--text_kes TEXT_KEYS] [--document_deduplicator CONFIG] [--document_deduplicator.hash_method HASH_METHOD] [--document_deduplicator.lowercase LOWERCASE]
                            [--document_deduplicator.ignore_non_character IGNORE_NON_CHARACTER] [--language_id_score_filter CONFIG] [--language_id_score_filter.lang LANG] [--words_num_filter CONFIG] [--words_num_filter.min MIN] [--words_num_filter.max MAX]
                            [--alphanumeric_filter CONFIG] [--alphanumeric_filter.min MIN] [--alphanumeric_filter.max MAX] [--average_line_length_filter CONFIG] [--average_line_length_filter.min MIN] [--average_line_length_filter.max MAX]
                            [--maximum_line_length_filter CONFIG] [--maximum_line_length_filter.min MIN] [--maximum_line_length_filter.max MAX] [--text_length_filter CONFIG] [--text_length_filter.min MIN] [--text_length_filter.max MAX]
                            [--remove_comments_mapper CONFIG] [--remove_comments_mapper.type TYPE] [--remove_comments_mapper.inline INLINE] [--remove_comments_mapper.multiline MULTILINE] [--remove_header_mapper CONFIG]
                            [--remove_header_mapper.before_section BEFORE_SECTION]

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.
  --project_name PROJECT_NAME
                        name of your data process project. (type: str, default: null)
  --dataset_path DATASET_PATH
                        path to your dataset file, relative with respect to the config file’s location (type: Path_fr, default: null)
  --dataset_dir DATASET_DIR
                        path to your dataset(s) within a directory, relative with respect to the config file’s location (type: Path_drw, default: null)
  --export_path EXPORT_PATH
                        path to the output processed dataset, relative with respect to the config file’s location (type: Path_fc, default: null)
  --process PROCESS, --process+ PROCESS
                        a list of several process operators with their arguments (type: List[Dict], default: null)
  --np NP               number of subprocess to process your dataset. (type: PositiveInt, default: null)

<class 'data_juicer.ops.filter.alphanumeric_filter.AlphanumericFilter'>:
  --alphanumeric_filter CONFIG
                        Path to a configuration file.
  --alphanumeric_filter.min MIN
                        the min filter rate in alphanumeric op. (type: ClosedUnitInterval, default: 0.0)
  --alphanumeric_filter.max MAX
                        the max filter rate in alphanumeric op. (type: ClosedUnitInterval, default: 0.25)

<class 'data_juicer.ops.filter.text_length_filter.TextLengthFilter'>:
  --text_length_filter CONFIG
                        Path to a configuration file.
  --text_length_filter.min MIN
                        min text length in the filtering (type: int, default: 10)
  --text_length_filter.max MAX
                        max text length in the filtering (type: int, default: 10000)

......

```
