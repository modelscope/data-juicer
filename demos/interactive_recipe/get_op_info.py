import sys
import os
import yaml
import inspect
from typing import get_origin, get_args, Union, Annotated, List, Dict, Optional, Tuple
from pydantic import PositiveInt, PositiveFloat
from jsonargparse.typing import ClosedUnitInterval
from loguru import logger

from data_juicer.tools.op_search import OPSearcher

MAX_NUM = 1000000
MIN_NUM = -1000000


def convert_union_int_tuple_or_none(x):
    if x is None or x == "None":
        return None
    elif isinstance(x, int):
        return x
    elif isinstance(x, list):
        return tuple(int(xx) for xx in x)
    else:
        raise ValueError("Unsupported type for Union[int, Tuple[int], Tuple[int, int], None]")


def safe_str_list_union(x):
    if isinstance(x, str):
        return x
    if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
        return [str(item) for item in x]
    return str(x)

def safe_int_str_union(x):
    if isinstance(x, (int, str)):
        return x
    try:
        return int(x)
    except (ValueError, TypeError):
        return str(x)

TYPE_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "Dict": dict,
    "List": list,
    "List[int]": lambda x: [int(xx) for xx in x],
    "List[str]": lambda x: [str(xx) for xx in x],
    "Optional[str]": lambda x: None if (x is None or x == "None") else str(x),
    "Optional[float]": lambda x: None if (x is None or x == "None") else float(x),
    "Optional[int]": lambda x: None if (x is None or x == "None") else int(x),
    "Optional[Dict]": lambda x: None if (x is None or x == "None") else dict(x),
    "Optional[List[str]]": lambda x: None if (x is None or x == "None") else [str(xx) for xx in x],
    "Optional[Tuple[int, int, int, int]]": lambda x: None if (x is None or x == "None") else (int(xx) for xx in x),
    "Union[int, Tuple[int], Tuple[int, int], None]": convert_union_int_tuple_or_none,
    "Union[str, List[str]]": safe_str_list_union,
    "Union[str, List[str], None]": safe_str_list_union,
    "Union[int, str]": safe_int_str_union,
    "Union[str, int, None]": lambda x: None if (x is None or x == "None") else safe_int_str_union(x),
}

TYPE_DEFINITIONS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "List[int]": List[int],
    "List[str]": List[str], 
    "Optional[str]": Optional[str],
    "Optional[int]": Optional[int],
    "Optional[List[str]]": Optional[List[str]],
    "Dict": Dict,
    "List": List,
    "Union[str, int, None]": Union[str, int, None],
    "Union[str, List[str]]": Union[str, List[str]],
    "Union[int, str]": Union[int, str],
    "Union[int, Tuple[int], Tuple[int, int], None]": Union[int, Tuple[int], Tuple[int, int], None],
}

def default_for_type(typ):
    """Given a type, returns common defaults for that type."""
    origin = get_origin(typ)
    args = get_args(typ)
    if typ in [str]:
        return ""
    elif typ in [int]:
        return 0
    elif typ in [float]:
        return 0.0
    elif typ in [bool]:
        return False
    elif origin in (list, List):
        return []
    elif origin in (dict, Dict):
        return {}
    elif origin in (tuple, Tuple):
        return ()
    elif origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return default_for_type(non_none_args[0])
        else:
            return None
    else:
        return None

def clean_text(text):
    """Clean text format"""
    if not text:
        return ""
    text = text.replace("\\\n", " ")
    text = text.replace("\\", " ")
    import re

    cleaned = re.sub(r"[\t\r\n\u3000 ]+", " ", text)
    return cleaned.strip()


def parse_type_annotation(annotation):
    """Parse type annotation to string representation and return constraint information"""
    if annotation == inspect.Parameter.empty:
        return "str", None, None  # default type, no constraints

    # Handle PositiveInt and PositiveFloat
    if annotation == PositiveInt:
        return "int", 1, None  # PositiveInt -> int, min=1 (actually > 0, but use 1 for UI)
    if annotation == PositiveFloat:
        return "float", 0.000001, None  # PositiveFloat -> float, min=very small positive number
    if annotation == ClosedUnitInterval:
        return "float", 0.0, 1.0


    # Handle basic types
    if annotation in (str, int, float, bool):
        type_str = annotation.__name__
        return type_str, None, None

    # Handle generic types (List, Optional, Union etc)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        # The first argument is the base type, the rest are metadata
        base_type = args[0]
        metadata = args[1:]
        
        # Recursively parse the base type to get its string representation
        type_str, _, _ = parse_type_annotation(base_type)
        
        # Initialize constraints
        min_val, max_val = None, None

        # Inspect metadata for constraints like ge, le, gt, lt
        for meta_item in metadata:
            # Pydantic v2 uses FieldInfo or constraints like Ge(), Le()
            # We can check for attributes directly for robustness
            if hasattr(meta_item, 'ge') and meta_item.ge is not None:
                min_val = meta_item.ge
            if hasattr(meta_item, 'gt') and meta_item.gt is not None:
                min_val = meta_item.gt
            if hasattr(meta_item, 'le') and meta_item.le is not None:
                max_val = meta_item.le
            if hasattr(meta_item, 'lt') and meta_item.lt is not None:
                max_val = meta_item.lt
            
            # Handle cases like FieldInfo(metadata=[Ge(0), Le(1)])
            if hasattr(meta_item, 'metadata'):
                for sub_meta in meta_item.metadata:
                    if hasattr(sub_meta, 'ge') and sub_meta.ge is not None:
                        min_val = sub_meta.ge
                    if hasattr(sub_meta, 'le') and sub_meta.le is not None:
                        max_val = sub_meta.le

        return type_str, min_val, max_val
    
    if origin is Union:
        # Handle Optional[T] (actually Union[T, None])
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            base_type, min_val, max_val = parse_type_annotation(non_none_type)
            return f"Optional[{base_type}]", min_val, max_val
        # Handle other Union types
        type_strs = []
        for arg in args:
            base_type, _, _ = parse_type_annotation(arg)
            type_strs.append(base_type)
        return f"Union[{', '.join(type_strs)}]", None, None

    if origin is list:
        if args:
            inner_type, _, _ = parse_type_annotation(args[0])
            return f"List[{inner_type}]", None, None
        return "List", None, None

    if origin is dict:
        return "Dict", None, None
    
    if origin is None:
        return "None", None, None
    
    if origin is tuple:
        if args:
            inner_types = []
            for arg in args:
                inner_type, _, _ = parse_type_annotation(arg)
                inner_types.append(inner_type)
            return f"Tuple[{', '.join(inner_types)}]", None, None

    # For other complex types, return string representation
    return str(annotation).replace("typing.", ""), None, None


def get_reasonable_limits(type_str, param_name, default_val):
    """Infer reasonable min/max values based on parameter type, name and default value"""
    if type_str not in ("int", "float"):
        return None, None

    # Default constraints
    if type_str == "int":
        return MIN_NUM, MAX_NUM
    else:  # float
        return float(MIN_NUM), float(MAX_NUM)


def extract_param_description(param_docstring, param_name):
    """Extract specific parameter description from parameter docstring"""
    if param_name == "trust_remote_code":
        return "Whether to trust the remote code for loading huggingface model."

    if not param_docstring or param_name not in param_docstring:
        return f"Parameter {param_name}"

    try:
        # Find parameter description
        param_section = param_docstring.split(f"{param_name}:")[1]
        # Find next parameter or end position
        next_param_pos = param_section.find(":param")
        if next_param_pos != -1:
            param_desc = param_section[:next_param_pos].strip()
        else:
            param_desc = param_section.strip()

        return clean_text(param_desc)
    except (IndexError, AttributeError):
        return f"Parameter {param_name}"


def convert_default_value(default_val, type_str):
    """Convert default value to appropriate type"""
    if default_val == inspect.Parameter.empty or default_val is None:
        # Provide reasonable default values based on type
        if "int" in type_str.lower():
            return 0
        elif "float" in type_str.lower():
            return 0.0
        elif "bool" in type_str.lower():
            return False
        elif "list" in type_str.lower():
            return []
        elif "dict" in type_str.lower():
            return {}
        else:
            return ""

    # Handle special values
    if (isinstance(default_val, int) or isinstance(default_val, float)) and default_val > MAX_NUM:
        return TYPE_MAPPING[type_str](MAX_NUM)
    
    print(default_val, type_str)

    return TYPE_MAPPING[type_str](default_val)


def extract_op_info_from_searcher(tags=None, op_type="filter"):
    """Extract operator information using OPSearcher"""
    # Only get filter type operators
    searcher = OPSearcher()
    filter_ops = searcher.search(tags=tags, op_type=op_type)

    all_op_info = {}

    for op_record in filter_ops:
        op_name = op_record["name"]
        logger.info(f"Processing op: {op_name}")

        # Handle special case reuse
        if op_name in ["llm_difficulty_score_filter", "llm_quality_score_filter"]:
            if "llm_analysis_filter" in all_op_info:
                all_op_info[op_name] = all_op_info["llm_analysis_filter"]
                continue

        if op_name == "video_motion_score_raft_filter":
            if "video_motion_score_filter" in all_op_info:
                all_op_info[op_name] = {
                    "desc": clean_text(op_record["desc"]),
                    "args": all_op_info["video_motion_score_filter"]["args"],
                }
                continue

        # Extract parameter information
        sig = op_record["sig"]
        param_docstring = op_record["param_desc"]

        params_info = {}

        # Iterate through parameters in signature (skip self)
        for param_name, param in sig.parameters.items():

            if param_name in ["self", "args", "kwargs"]:
                continue

            # Parse parameter type
            type_str, constraint_min, constraint_max = parse_type_annotation(param.annotation)

            if param_name == "trust_remote_code":
                type_str = "bool"

            # Get default value
            default_val = convert_default_value(param.default, type_str)

            # Extract parameter description
            param_desc = extract_param_description(param_docstring, param_name)

            param_info = {
                "desc": param_desc,
                "type": type_str,
                "default": default_val,
            }

            limit_min, limit_max = get_reasonable_limits(type_str, param_name, default_val)

            if type_str in ("int", "float"):
                if constraint_min is not None:
                    param_info["min"] = constraint_min
                elif limit_min is not None:
                    param_info["min"] = limit_min
                if constraint_max is not None:
                    param_info["max"] = constraint_max
                elif limit_max is not None:
                    param_info["max"] = limit_max
            
            if param_name == 'any_or_all':
                param_info["options"] = ["any", "all"]

            params_info[param_name] = param_info

        all_op_info[op_name] = {
            "desc": clean_text(op_record["desc"]),
            "args": params_info,
        }

    return all_op_info


if __name__ == "__main__":
    all_op_info = extract_op_info_from_searcher(op_type=None)
    file_path = os.path.join(os.path.dirname(__file__), "configs", "all_op_info.yaml")

    with open(file_path, "w") as f:
        yaml.safe_dump(all_op_info, f, sort_keys=False)

    logger.info(f"Successfully processed {len(all_op_info)} operators")