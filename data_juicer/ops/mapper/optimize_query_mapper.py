from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.optimize_qa_mapper import OptimizeQAMapper

OP_NAME = "optimize_query_mapper"


# TODO: Extend LLM-based OPs into API-based implementation.
@OPERATORS.register_module(OP_NAME)
class OptimizeQueryMapper(OptimizeQAMapper):
    """
    Mapper to optimize query in question-answer pairs.
    """

    DEFAULT_SYSTEM_PROMPT = "优化问答对中的【问题】，将其更加详细具体，但仍可以由原答案回答。只输出优化后的【问题】，不要输出多余内容。"  # noqa: E501

    _accelerator = "cuda"

    def parse_output(self, raw_output):
        return raw_output.strip(), None
