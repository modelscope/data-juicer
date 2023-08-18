# 开发者指南

* [开发者指南](#开发者指南)
   * [编码规范](#编码规范)
   * [构建自己的算子](#构建自己的算子)
   * [构建自己的配置](#构建自己的配置)
      * [丰富的配置源和类型提示](#丰富的配置源和类型提示)
      * [层次化的配置和帮助](#层次化的配置和帮助)

## 编码规范

我们将编码规范定义在 `.pre-commit-config.yaml` 中。在向仓库贡献代码之前，请使用 `pre-commit` 工具对代码进行规范化。

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

3. 实现后，将其添加到 `data_juicer/ops/filter` 目录下 `__init__.py` 文件中的算子字典中：

```python
from . import (...,              # other ops
               text_length_filter)  # import this new op module

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

class TextLengthFilterTest(unittest.TestCase):

    def test_func1(self):
        pass

    def test_func2(self):
        pass

    def test_func3(self):
        pass
```

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
