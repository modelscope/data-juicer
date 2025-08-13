from data_juicer.utils.registry import Registry

ALL_FUNCS = Registry("all_helper_funcs")


# LLM inference funcs
@ALL_FUNCS.register_module("build_messages")
def build_messages(item: dict, **kwargs):
    """
    A simple implementation.
    """
    system_key = kwargs.get("system_key", "system")
    query_key = kwargs.get("query_key", "query")

    system_prompt = item.get(system_key, "")
    input_content = item[query_key]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_content})
    return messages


@ALL_FUNCS.register_module("parse_output")
def parse_output(output: str, item: dict, **kwargs):
    """
    A simple implementation.
    """
    return output
