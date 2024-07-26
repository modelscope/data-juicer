# How-to Guide for Developers

- [How-to Guide for Developers](#how-to-guide-for-developers)
  - [Coding Style](#coding-style)
  - [Build your own OPs](#build-your-own-ops)
    - [(Optional) Make your OP fusible](#optional-make-your-op-fusible)
  - [Build your own configs](#build-your-own-configs)
    - [Fruitful config sources \& Type hints](#fruitful-config-sources--type-hints)
    - [Hierarchical configs and helps](#hierarchical-configs-and-helps)

## Coding Style

We define our styles in `.pre-commit-config.yaml`. Before committing,
please install `pre-commit` tool to automatically check and modify accordingly:

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

**Note**: We have configured pre-commit checks in github workflow. If this 
check in your PR fails, please locally ① ensure that the relevant 
dependencies of pre-commit are consistent with the project configuration 
(which can be completed through `pre-commit clean` and `pre-commit install`); 
and ② execute `pre-commit run --all-files` before push.

## Build your own OPs

- Data-Juicer allows everybody to build their own OPs.
- Before implementing a new OP, please refer to [Operators](Operators.md) to avoid unnecessary duplication.
- Assuming we want to add a new Filter operator called "TextLengthFilter" to get corpus of expected text length, we can follow these steps to build it.

1. (Optional) Add a new StatsKeys in `data_juicer/utils/constant.py` to store the statistical variable of the new OP.

```python
class StatsKeys(object):
    ...              # other keys
    text_len = 'text_len'
```

2. Create a new OP file `text_length_filter.py` in the corresponding `data_juicer/ops/filter/` directory as follows.
   - Because it's a Filter OP, so the new OP needs to inherit from the basic `Filter` class in the `base_op.py`, and be decorated with `OPERATORS` to register itself automatically.

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

    - If Hugging Face models are used within an operator, you might want to leverage GPU acceleration. To achieve this, declare `_accelerator = 'cuda'` in the constructor, and ensure that `compute_stats` and `process` methods accept an additional positional argument `rank`.

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

    - If the operator processes data in batches rather than a single sample, it is necessary to declare `_batched_op = True`.
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

    - In a mapper operator, to avoid process conflicts and data coverage, we offer an interface to make a saving path for produced extra datas. The format of the saving path is `{ORIGINAL_DATAPATH}/__dj__produced_data__/{OP_NAME}/{ORIGINAL_FILENAME}__dj_hash_#{HASH_VALUE}#.{EXT}`, where the `HASH_VALUE` is hashed from the init parameters of the operator, the related parameters in each sample, the process ID, and the timestamp. For convenience, we can call `self.remove_extra_parameters(locals())` at the beginning of the initiation to get the init parameters. At the same time, we can call `self.add_parameters` to add related parameters with the produced extra datas from each sample. Take the operator which enhances the images with diffusion models as example:
    ```python
    from data_juicer.utils.file_utils import transfer_filename
    # ... (import some other libraries)
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
    For the mapper to produce multi extra datas base on one origin data, we can add suffix at the saving path. Take the operator which splits videos according to their key frames as example:
    ```python
    from data_juicer.utils.file_utils import add_suffix_to_filename, transfer_filename
    # ... (import some other libraries)
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
            split_video_path = add_suffix_to_filename(split_video_path,  f'_{count}')
            # ... (some codes)
    ```

3. After implemention, add it to the OP dictionary in the `__init__.py` file in `data_juicer/ops/filter/` directory.

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

4. Now you can use this new OP with custom arguments in your own config files!

```yaml
# other configs
...

# process configs
process:
  - text_length_filter:  # add this OP to your process list and set the parameters
      min_len: 10
      max_len: 1000
```

5. (Strongly Recommend) It's better to add corresponding tests for your own OPs. For `TextLengthFilter` above, you would like to add `test_text_length_filter.py` into `tests/ops/filter/` directory as below.

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

6. (Strongly Recommend) In order to facilitate the use of other users, we also need to update this new OP information to
the corresponding documents, including the following docs:
   1. `configs/config_all.yaml`: this complete config file contains a list of all OPs and their arguments, serving as an
   important document for users to refer to all available OPs. Therefore, after adding the new OP, we need to add it to the process
   list (grouped by the OP type and sorted in alphabetical order):
   
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
   
   2. `docs/Operators.md`: this doc maintains categorized lists of available OPs. We can add the information of new OP to the list
   of corresponding type of OPs (sorted in alphabetical order). At the same time, in the Overview section at the top of this doc,
   we also need to update the number of OPs for the corresponding OP type:

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

   3. `docs/Operators_ZH.md`: this doc is the Chinese version of the doc in 6.ii, so we need to update the Chinese content at
   the same positions.


### (Optional) Make your OP fusible

- If the calculation process of some intermediate variables in the new OP is reused in other existing OPs, this new OP can be
added to the fusible OPs to accelerate the whole data processing with OP fusion technology. (e.g. both the `words_num_filter`
and `word_repetition_filter` need to split the input text into words)
- When opening OP fusion, these reused calculation processes and intermediate variables can be shared in the `context` between
OPs, thus reducing repeated calculations.
- OPs that contain common intermediate variables can be fused in OP fusion through the following steps:

1. (Optional) If a new intermediate variable is generated in the new OP, we need to add this new intermediate variable name to 
the `InterVars` class in `utils/constant.py`. In general, we need to add a prefix `DEFAULT_PREFIX` before the name.

```python
class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'  # add the new intermediate variable here
    ...
```

2. (Optional) We need to define a registry group in `ops/op_fusion.py` for the new intermediate variable in the 1st step, and add
this registry group to the registry group list that stores all groups of intermediate variables. This facilitates the OP Fusion module
to track OPs involving these intermediate variables.

```python
...
# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)  # define registry group for the new intermediate variable

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# all
ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_IMAGES]  # and add it to the registry group list
...
```

3. Before the OP class definition that involves the intermediate variable, register this OP in the registry group corresponding
to this intermediate variable, indicating that the intermediate variable may be calculated and used in this OP.

```python
...
@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)  # register this new OP into the registry group
class WordsNumFilter(Filter):
...
```

4. In the calculation process of this intermediate variable of the new OP, we can modify the calculation logic to:
   1. If the argument `context` is True, it means the OP fusion is opening, so we get the value of this intermediate variable 
   from `context` first, which has been calculated by the previous OPs.
   2. If this intermediate variable doesn't exist in the `context`, it means it's the first time to calculate this variable in this
   OP, so we need to define a unique key and use it to store the intermediate variable in the `context` for subsequent OPs after
   it's calculated by this new OP.
   3. If the argument `context` is False, just follow the normal calculation process.

```python
# before modification
...
tokenizer = get_model(self.model_key)
words = get_words_from_document(
    sample[self.text_key],
    token_func=tokenizer.encode_as_pieces if tokenizer else None)
...        

# after modification
...
words_key = f'{InterVars.words}-{self.model_key}'
if context and words_key in sample[Fields.context]:
    # get the value of intermediate variable from context directly
    words = sample[Fields.context][words_key]
else:
    # normal calculation process
    tokenizer = get_model(self.model_key)
    words = get_words_from_document(
        sample[self.text_key],
        token_func=tokenizer.encode_as_pieces if tokenizer else None)
    if context:
        # After calculating the intermediate variable for the first time,
        # store it in the context for subsequent OPs.
        sample[Fields.context][words_key] = words
...
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
