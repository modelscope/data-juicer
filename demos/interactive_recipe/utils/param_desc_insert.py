import re
from pathlib import Path

from docstring_parser import parse

ROOT = Path(__file__).resolve().parents[3]
OPS_DIR = ROOT / "data_juicer" / "ops"


def find_op_file(op_name):
    """Locate the corresponding operator file for a given operator name."""
    pattern = f"{op_name}.py"
    matches = list(OPS_DIR.rglob(pattern))
    return matches[0] if matches else None


def extract_init_docstring(content):
    """Extract the __init__ method's docstring from file content."""
    # Pattern to match __init__ method with its docstring
    pattern = r'def __init__\([\s\S]*?\):\s*"""([\s\S]*?)"""'
    match = re.search(pattern, content)

    if match:
        docstring_content = match.group(1)
        full_docstring = f'"""{docstring_content}"""'
        return docstring_content.strip(), full_docstring
    return None, None


def get_init_params_order(content):
    """Extract parameter order from __init__ method signature."""
    pattern = r"def __init__\(\s*([\s\S]*?)\s*\):"
    match = re.search(pattern, content)

    if not match:
        return []

    signature = match.group(1)
    # Split by comma and clean up parameter names
    params = []
    for param in signature.split(","):
        param = param.strip()
        if param and param != "self" and not param.startswith("*"):
            # Extract parameter name (before : or =)
            param_name = param.split(":")[0].split("=")[0].strip()
            if param_name:
                params.append(param_name)

    return params


def format_param_description(param_name, description):
    """Format parameter description with proper indentation for multi-line."""
    if not description:
        return f":param {param_name}:"

    # Split description into lines
    lines = description.strip().split("\n")
    if len(lines) == 1:
        return f":param {param_name}: {lines[0]}"

    # Multi-line description
    result = [f":param {param_name}: {lines[0]}"]
    for line in lines[1:]:
        # Add proper indentation for continuation lines
        result.append(f"            {line.strip()}")

    return "\n".join(result)


def reconstruct_docstring(parsed_docstring, param_order, new_param_desc):
    """Reconstruct docstring with proper parameter order and indentation."""
    # Start with short and long description
    lines = []
    if parsed_docstring.short_description:
        lines.append(parsed_docstring.short_description)

    if parsed_docstring.long_description:
        if lines:  # Add blank line if there's a short description
            lines.append("")
        # Handle multi-line long description
        long_desc_lines = parsed_docstring.long_description.strip().split("\n")
        lines.extend(long_desc_lines)

    # Add blank line before parameters if there are descriptions
    if lines:
        lines.append("")

    # Create parameter map from existing docstring
    existing_params = {param.arg_name: param for param in parsed_docstring.params}

    # Add parameters in the order they appear in __init__ signature
    param_lines = []
    for param_name in param_order:
        if param_name in existing_params:
            param = existing_params[param_name]
            formatted_param = format_param_description(param.arg_name, param.description)
            param_lines.append(formatted_param)
        elif param_name == "trust_remote_code":
            # Add the new trust_remote_code parameter
            param_lines.append(new_param_desc.rstrip())

    # Add any remaining parameters that weren't in the signature order
    for param in parsed_docstring.params:
        if param.arg_name not in param_order:
            formatted_param = format_param_description(param.arg_name, param.description)
            param_lines.append(formatted_param)

    # Join parameter lines and add to main lines
    if param_lines:
        lines.extend(param_lines)

    # Add other sections (returns, raises, etc.)
    if parsed_docstring.returns:
        lines.append("")
        return_desc = parsed_docstring.returns.description or ""
        if return_desc:
            return_lines = return_desc.strip().split("\n")
            lines.append(f":return: {return_lines[0]}")
            for line in return_lines[1:]:
                lines.append(f"            {line.strip()}")
        else:
            lines.append(":return:")

    if parsed_docstring.raises:
        lines.append("")
        for exc in parsed_docstring.raises:
            exc_type = exc.type_name or ""
            exc_desc = exc.description or ""
            if exc_desc:
                exc_lines = exc_desc.strip().split("\n")
                lines.append(f":raises {exc_type}: {exc_lines[0]}" if exc_type else f":raises: {exc_lines[0]}")
                for line in exc_lines[1:]:
                    lines.append(f"            {line.strip()}")
            else:
                lines.append(f":raises {exc_type}:" if exc_type else ":raises:")

    strings = ""
    for i, line in enumerate(lines):
        if i == 0:
            strings += line
            continue
        if line:
            strings += "\n        " + line
        else:
            strings += "\n"
    return strings


def insert_missing_param_desc(op_name, missing_param):
    """Insert missing parameter description for an operator."""
    op_file = find_op_file(op_name)
    if not op_file:
        print(f"Could not find file for operator: {op_name}")
        return None

    # Read the file content
    with open(op_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the operator is registered
    if "@OPERATORS.register_module" not in content:
        print(f"Operator {op_name} is not registered with @OPERATORS.register_module")
        return None

    # Check if missing parameter exists
    if missing_param not in content:
        print(f"missing parameter not found in {op_name}")
        return None

    # Extract docstring
    docstring_content, full_docstring = extract_init_docstring(content)
    if not docstring_content:
        print(f"Could not find __init__ docstring in {op_name}")
        return None

    # Parse the docstring
    try:
        parsed_docstring = parse(docstring_content)
    except Exception as e:
        print(f"Failed to parse docstring in {op_name}: {e}")
        return None

    # Check if missing parameter description already exists
    existing_param_names = [param.arg_name for param in parsed_docstring.params]
    if missing_param in existing_param_names:
        print(f"{missing_param} description already exists in {op_name}")
        return None

    # Get parameter order from __init__ signature
    param_order = get_init_params_order(content)
    if missing_param not in param_order:
        print(f"{missing_param} not found in __init__ signature of {op_name}")
        return None

    # Define the new parameter description
    if missing_param == "trust_remote_code":
        new_param_desc = ":param trust_remote_code: whether to trust the remote code of HF models."
    else:
        # TODO: generate new parameter description by LLM
        new_param_desc = ""

    # Reconstruct the docstring
    new_docstring_content = reconstruct_docstring(parsed_docstring, param_order, new_param_desc)

    # Create the new full docstring
    new_full_docstring = f'"""\n        {new_docstring_content}\n        """'

    # Replace in the content
    new_content = content.replace(full_docstring, new_full_docstring)

    # Write back to file
    with open(op_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully inserted trust_remote_code description in {op_name}")
    return new_param_desc


def batch_insert_trust_remote_code_desc(op_names):
    """Batch insert trust_remote_code descriptions for multiple operators."""
    success_count = 0
    for op_name in op_names:
        try:
            if insert_missing_param_desc(op_name, "trust_remote_code"):
                success_count += 1
        except Exception as e:
            print(f"Error processing {op_name}: {e}")

    print(f"Successfully processed {success_count}/{len(op_names)} operators")


if __name__ == "__main__":
    missing_ops = [
        "image_captioning_mapper",
        "image_diffusion_mapper",
        "video_watermark_filter",
    ]

    batch_insert_trust_remote_code_desc(missing_ops)
