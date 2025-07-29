from data_juicer.utils.registry import Registry

ALL_FUNCS = Registry("all_helper_funcs")


# LLM inference funcs
@ALL_FUNCS.register_module("build_messages")
def build_messages(item: dict):
    """
    A simple implementation.
    """
    system_prompt = item.get("system", "")
    input_content = item["query"]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_content})
    return messages


@ALL_FUNCS.register_module("parse_output")
def parse_output(output: str, item: dict):
    """
    A simple implementation.
    """
    return output
