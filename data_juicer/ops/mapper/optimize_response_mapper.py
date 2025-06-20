from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.optimize_qa_mapper import OptimizeQAMapper

OP_NAME = "optimize_response_mapper"


# TODO: Extend LLM-based OPs into API-based implementation.
@OPERATORS.register_module(OP_NAME)
class OptimizeResponseMapper(OptimizeQAMapper):
    """
    Mapper to optimize response in question-answer pairs.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "请优化问答对中的回答，将其更加详细具体，但仍可以回答原问题。只输出优化后的回答，不要输出多余内容。"
    )

    _accelerator = "cuda"

    def parse_output(self, raw_output):
        return None, raw_output.strip()
