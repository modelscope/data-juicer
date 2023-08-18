# How-to Guide for Developers

* [How-to Guide for Developers](#how-to-guide-for-developers)
   * [Coding Style](#coding-style)
   * [Build your own ops](#build-your-own-ops)
   * [Build your own configs](#build-your-own-configs)
      * [Fruitful config sources &amp; Type hints](#fruitful-config-sources--type-hints)
      * [Hierarchical configs and helps](#hierarchical-configs-and-helps)

## Coding Style

We define our styles in `.pre-commit-config.yaml`. Before committing,
please install `pre-commit` tool to check and modify accordingly:

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
git commit -m "xxxx"
```

## Build your own ops

- Data-Juicer allows everybody to build their own ops.
- Before implementing a new op, please refer to [Operators](Operators.md) to avoid unnecessary duplication.
- Assuming we want to add a new Filter operator called "TextLengthFilter" to get corpus of expected text length, we can follow these steps to build it.

1. (Optional) Add a new StatsKeys in `data_juicer/utils/constant.py` to store the statistical variable of the new op.

```python
class StatsKeys(object):
    ...              # other keys
    text_len = 'text_len'
```

2. Create a new op file `text_length_filter.py` in the corresponding `data_juicer/ops/filter/` directory as follows.
   - Because it's a Filter op, so the new op needs to inherit from the basic `Filter` class in the `base_op.py`, and be decorated with `OPERATORS` to register itself automatically.

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

3. After implemention, add it to the op dictionary in the `__init__.py` file in `data_juicer/ops/filter/` directory.

```python
from . import (...,              # other ops
               text_length_filter)  # import this new op module
```

4. Now you can use this new op with custom arguments in your own config files!

```yaml
# other configs
...

# process configs
process:
  - text_length_filter:  # add this op to your process list and set the parameters
      min_len: 10
      max_len: 1000
```

5. (Strongly Recommend) It's better to add corresponding tests for your own ops. For `TextLengthFilter` above, you would like to add `test_text_length_filter.py` into `tests/ops/filter/` directory as below.

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

## Build your own configs
- We provide easy configuration based on [jsonargparse](https://github.com/omni-us/jsonargparse/) to reduce cost for boilerplate codes.

### Fruitful config sources & Type hints
- A global config object can be initialized via
```
# core.executor.py
self.cfg = init_configs()
```
- in which function arguments from diverse sources can be specified and mixed
up, including
1. *hard-coded default values* when registering the config into parser or specified in the classes' `__init__` functions
2. default *config files* in json (yaml or jsonnet supersets)
3. *environment variables*
4. *POSIX-style command line arguments*, such as ``--project_name
   my_data_demo`` or ``--project_name=my_data_demo`` , including config files

- The final parsed values are mixed from these sources. And the override order is the same as the numbers above.

Besides, many argument types and respective validation are supported.
Including python built-in types, types from [Lib/typing](https://docs.python.org/3/library/typing.html) module, and
extended [types](https://jsonargparse.readthedocs.io/en/stable/#type-hints)
from jsonargparse, such as `restricted types` and `Paths` with customized
limitations.

### Hierarchical configs and helps
- You can use dot notation in the argument names freely to define the
hierarchy, e.g., `maximum_line_length_filter.min`.
More importantly, by default, we automatically register the configs from
the docstrings of implemented operators. That is, the structure of all
configs are always in sync with codes.

- You can get the hierarchical help information by running a script that calls
our executor such as
```
$ python tools/process_data.py --help

usage: process_data.py [-h] [--config CONFIG] [--print_config[=flags]] [--project_name PROJECT_NAME] [--dataset_path DATASET_PATH] [--dataset_dir DATASET_DIR] [--export_path EXPORT_PATH] [--process PROCESS]
                            [--np NP] [--text_keys TEXT_KEYS] [--document_deduplicator CONFIG] [--document_deduplicator.hash_method HASH_METHOD] [--document_deduplicator.lowercase LOWERCASE]
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
