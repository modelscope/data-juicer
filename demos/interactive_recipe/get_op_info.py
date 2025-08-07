import sys
import os
import yaml
import inspect
from loguru import logger

from data_juicer.tools.op_search import OPSearcher
from utils.param_type_utils import TypeAnnotationParser

# Constants
MAX_NUM = 1000000
MIN_NUM = -1000000


class ParameterProcessor:
    """Process function parameters and extract metadata."""

    def __init__(self):
        self.parser = TypeAnnotationParser()

    def convert_default_value(self, default_val, typ):
        """Convert default value to appropriate type."""
        if default_val == inspect.Parameter.empty:
            return self.parser.get_default_value(typ)
        
        if default_val is None:
            import typing
            origin = typing.get_origin(typ)
            args = typing.get_args(typ)
            if origin is typing.Union:
                if type(None) in args:
                    return None
            else:
                return self.parser.get_default_value(typ)

        # Handle special large values
        if isinstance(default_val, (int, float)) and default_val > MAX_NUM:
            converter = self.parser.converter(typ)
            return converter(MAX_NUM)

        converter = self.parser.converter(typ)
        return converter(default_val)

    def get_reasonable_limits(self, type_str):
        """Get reasonable min/max limits for numeric types."""
        if type_str not in ("int", "float", "Optional[int]", "Optional[float]"):
            return None, None

        if type_str in ("int", "Optional[int]"):
            return MIN_NUM, MAX_NUM
        return float(MIN_NUM), float(MAX_NUM)

    def extract_param_description(self, param_docstring, param_name):
        """Extract parameter description from docstring."""
        if param_name == "trust_remote_code":
            return "Whether to trust the remote code for loading huggingface model."

        if not param_docstring or param_name not in param_docstring:
            raise ValueError(f"Failed to extract parameter description for {param_name}")

        try:
            # Find parameter description
            param_section = param_docstring.split(f"{param_name}:")[1]
            next_param_pos = param_section.find(":param")
            if next_param_pos != -1:
                param_desc = param_section[:next_param_pos].strip()
            else:
                param_desc = param_section.strip()

            return self._clean_text(param_desc)
        except (IndexError, AttributeError):
            raise ValueError(f"Failed to extract parameter description for {param_name}")

    def _clean_text(self, text):
        """Clean and format text."""
        if not text:
            return ""

        text = text.replace("\\\n", " ").replace("\\", " ")

        import re

        cleaned = re.sub(r"[\t\r\n\u3000 ]+", " ", text)
        return cleaned.strip()


class OperatorInfoExtractor:
    """Extract operator information from OPSearcher."""

    def __init__(self):
        self.parser = TypeAnnotationParser()
        self.processor = ParameterProcessor()

    def extract_all_operators(self, tags=None, op_type="filter"):
        """Extract information for all operators matching criteria."""
        searcher = OPSearcher()
        filter_ops = searcher.search(tags=tags, op_type=op_type)

        all_op_info = {}

        for op_record in filter_ops:
            op_name = op_record["name"]
            logger.info(f"Processing operator: {op_name}")

            # Handle special case reuse
            if self._should_reuse_config(op_name, all_op_info):
                all_op_info[op_name] = self._get_reused_config(op_name, all_op_info, op_record)
                continue

            # Extract parameter information
            op_info = self._extract_single_operator(op_record)
            all_op_info[op_name] = op_info

        return all_op_info

    def _should_reuse_config(self, op_name, all_op_info):
        """Check if operator should reuse existing configuration."""
        reuse_map = {
            "llm_difficulty_score_filter": "llm_analysis_filter",
            "llm_quality_score_filter": "llm_analysis_filter",
            "video_motion_score_raft_filter": "video_motion_score_filter",
        }

        return op_name in reuse_map and reuse_map[op_name] in all_op_info

    def _get_reused_config(self, op_name, all_op_info, op_record):
        """Get reused configuration for special operators."""
        if op_name in ["llm_difficulty_score_filter", "llm_quality_score_filter"]:
            return all_op_info["llm_analysis_filter"]

        if op_name == "video_motion_score_raft_filter":
            return {
                "desc": self.processor._clean_text(op_record["desc"]),
                "args": all_op_info["video_motion_score_filter"]["args"],
            }

        return {}

    def _extract_single_operator(self, op_record):
        """Extract information for a single operator."""
        sig = op_record["sig"]
        param_docstring = op_record["param_desc"]
        params_info = {}

        # Process each parameter
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "args", "kwargs"]:
                continue

            param_info = self._process_parameter(param_name, param, param_docstring)
            params_info[param_name] = param_info

        return {
            "desc": self.processor._clean_text(op_record["desc"]),
            "args": params_info,
        }

    def _process_parameter(self, param_name, param, param_docstring):
        """Process a single parameter and extract its metadata."""
        # Parse parameter type
        type_str, constraint_min, constraint_max = self.parser.parse_annotation(param.annotation)

        # Handle special cases
        if param_name == "trust_remote_code":
            type_str = "bool"

        # Infer type from default value if not available
        if type_str is None and param.default not in (None, inspect.Parameter.empty):
            type_str, _, _ = self.parser.parse_annotation(type(param.default))

        if type_str is None:
            type_str = "str"

        # Get default value
        default_val = self.processor.convert_default_value(param.default, self.parser.str_to_type(type_str))

        # Extract parameter description
        param_desc = self.processor.extract_param_description(param_docstring, param_name)

        # Build parameter info
        param_info = {
            "desc": param_desc,
            "type": type_str,
            "default": default_val,
        }

        # Add numeric constraints
        if type_str in ("int", "float", "Optional[int]", "Optional[float]"):
            limit_min, limit_max = self.processor.get_reasonable_limits(type_str)
            param_info["min"] = constraint_min if constraint_min is not None else limit_min
            param_info["max"] = constraint_max if constraint_max is not None else limit_max

        # Add special options
        if param_name == "any_or_all":
            param_info["options"] = ["any", "all"]

        return param_info


def main():
    """Main function to extract and save operator information."""
    extractor = OperatorInfoExtractor()
    all_op_info = extractor.extract_all_operators(op_type=None)

    file_path = os.path.join(os.path.dirname(__file__), "configs", "all_op_info.yaml")

    with open(file_path, "w") as f:
        yaml.safe_dump(all_op_info, f, sort_keys=False)

    logger.info(f"Successfully processed {len(all_op_info)} operators")


if __name__ == "__main__":
    main()
